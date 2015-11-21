#include "conditioned_sfs.h"

std::map<int, below_coeff> below_coeffs_memo;
below_coeff compute_below_coeffs(int n)
{
    if (below_coeffs_memo.count(n) == 0)
    {
        PROGRESS("Computing below_coeffs");
        below_coeff ret;
        MatrixXq mlast;
        for (int nn = 2; nn < n + 3; ++nn)
        {
            MatrixXq mnew(n + 1, nn - 1);
            mnew.col(nn - 2).setZero();
            mnew(nn - 2, nn - 2) = 1_mpq;
            for (int k = nn - 1; k > 1; --k)
            {
                long denom = (nn + 1) * (nn - 2) - (k + 1) * (k - 2);
                MPQ_CONSTRUCT(c1, (nn + 1) * (nn - 2), denom);
                MPQ_CONSTRUCT(c2, (k + 2) * (k - 1), denom);
                mnew.col(k - 2) = mlast.col(k - 2) * c1;
                mnew.col(k - 2) -= mnew.col(k - 1) * c2;
            }
            mlast = mnew;
        }
        ret.coeffs = mlast;
        below_coeffs_memo.emplace(n, ret); 
        /*
        for (int i = 0; i < ret.coeffs.rows(); ++i)
        {
            std::cout << "{";
            for (int j = 0; j < ret.coeffs.cols(); ++j)
            {
                std::cout << ret.coeffs(i, j);
                if (j < ret.coeffs.cols() - 1)
                    std::cout << ",";
            }
            std::cout << "},\n";
        }
        std::cout << std::endl;
        */
        PROGRESS_DONE();
    }
    return below_coeffs_memo[n];
}

std::map<std::array<int, 3>, mpq_class> _Wnbj_memo;
mpq_class calculate_Wnbj(int n, int b, int j)
{
    switch (j)
    {
        case 2:
            return mpq_class(6, n + 1);
        case 3:
            if (n == 2 * b) return 0_mpq;
            return mpq_class(30 * (n - 2 * b), (n + 1) * (n + 2));
        default:
            std::array<int, 3> key = {n, b, j};
            if (_Wnbj_memo.count(key) == 0)
            {
                int jj = j - 2;
                MPQ_CONSTRUCT(c1, -(1 + jj) * (3 + 2 * jj) * (n - jj), jj * (2 * jj - 1) * (n + jj + 1));
                MPQ_CONSTRUCT(c2, (3 + 2 * jj) * (n - 2 * b), jj * (n + jj + 1));
                mpq_class ret = calculate_Wnbj(n, b, jj) * c1;
                ret += calculate_Wnbj(n, b, jj + 1) * c2;
                _Wnbj_memo[key] = ret;
            }
            return _Wnbj_memo[key];
    }
}

std::map<std::array<int, 3>, mpq_class> pnkb_dist_memo;
mpq_class pnkb_dist(int n, int m, int l1)
{
    // Probability that lineage 1 has size |L_1|=l1 below tau,
    // the time at which 1 and 2 coalesce, when there are k 
    // undistinguished lineages remaining, in a sample of n
    // undistinguished (+2 distinguished) lineages overall.
    std::array<int, 3> key{n, m, l1};
    if (pnkb_dist_memo.count(key) == 0)
    {
        mpz_class binom1, binom2;
        mpz_bin_uiui(binom1.get_mpz_t(), n + 2 - l1, m + 1);
        mpz_bin_uiui(binom2.get_mpz_t(), n + 3, m + 3);
        MPQ_CONSTRUCT(ret, binom1, binom2);
        ret *= l1;
        pnkb_dist_memo[key] = ret;
    }
    return pnkb_dist_memo[key];
}

std::map<std::array<int, 3>, mpq_class> pnkb_undist_memo;
mpq_class pnkb_undist(int n, int m, int l3)
{
    // Probability that undistinguished lineage has size |L_1|=l1 below tau,
    // the time at which 1 and 2 coalesce, when there are k 
    // undistinguished lineages remaining, in a sample of n
    // undistinguished (+2 distinguished) lineages overall.
    std::array<int, 3> key{n, m, l3};
    if (pnkb_undist_memo.count(key) == 0)
    {
        mpz_class binom1, binom2;
        mpz_bin_uiui(binom1.get_mpz_t(), n + 3 - l3, m + 2);
        mpz_bin_uiui(binom2.get_mpz_t(), n + 3, m + 3);
        MPQ_CONSTRUCT(ret, binom1, binom2);
        pnkb_undist_memo.emplace(key, ret);
    }
    return pnkb_undist_memo[key];
}

template <typename T>
ConditionedSFS<T>::ConditionedSFS(int n, int H) :
    n(n), H(H),
    mei(compute_moran_eigensystem(n)), mcache(cached_matrices(n)),
    tjj_below(H, n + 1),
    M0_below(H, n), M1_below(H, n + 1),
    csfs(H, Matrix<T>::Zero(3, n + 1)), csfs_below(H, Matrix<T>::Zero(3, n + 1)), csfs_above(H, Matrix<T>::Zero(3, n + 1)),
    C_above(H, Matrix<T>::Zero(n + 1, n))
{}

template <typename T>
void ConditionedSFS<T>::compute_below(const PiecewiseExponentialRateFunction<T> &eta)
{
    PROGRESS("compute below");
    Matrix<T> ts_integrals(eta.K, n + 1); 
#pragma omp parallel for
    for (int m = 0; m < eta.K; ++m)
        eta.tjj_double_integral_below(n, m, ts_integrals);
    Matrix<T> last = ts_integrals.topRows(eta.hs_indices[0]).colwise().sum(), next;
    for (int h = 1; h < H + 1; ++h)
    {
        next = ts_integrals.topRows(eta.hs_indices[h]).colwise().sum();
        tjj_below.row(h - 1) = next - last;
        last = next;
    }
    PROGRESS("matrix products below");
    M0_below = tjj_below * mcache.M0.template cast<T>();
    M1_below = tjj_below * mcache.M1.template cast<T>();
    PROGRESS("mpfr sfs below");
    for (int h = 0; h < H; ++h) 
    {
        csfs_below[h].fill(eta.zero);
        csfs_below[h].block(0, 1, 1, n) = M0_below.row(h);
        csfs_below[h].block(1, 0, 1, n + 1) = M1_below.row(h);
        check_nan(csfs_below[h]);
    }
    PROGRESS_DONE();
}

template <typename T>
void ConditionedSFS<T>::compute_above(const PiecewiseExponentialRateFunction<T> &eta)
{
    PROGRESS("compute above");
    PROGRESS("tjj double integral");
#pragma omp parallel for
    for (int j = 2; j < n + 3; ++j)
        eta.tjj_double_integral_above(n, j, C_above);
    Matrix<T> tmp;
    PROGRESS("matrix products");
    MatrixXq Uinv_mp0 = mei.Uinv.rightCols(n), Uinv_mp2 = mei.Uinv.reverse().leftCols(n);
#pragma omp parallel for
    for (int h = 0; h < H; ++h)
    {
        csfs_above[h].fill(eta.zero);
        csfs_above[h].block(0, 1, 1, n) = 
            mcache.X0.template cast<T>().cwiseProduct(C_above[h].transpose()).colwise().sum() * Uinv_mp0.template cast<T>();
        csfs_above[h].block(2, 0, 1, n) = 
            mcache.X2.template cast<T>().cwiseProduct(C_above[h].colwise().reverse().transpose()).colwise().sum() * Uinv_mp2.template cast<T>();
        check_nan(csfs_above[h]);
    }
    PROGRESS_DONE();
}

template <typename T>
std::vector<Matrix<T> >& ConditionedSFS<T>::compute(const PiecewiseExponentialRateFunction<T> &eta, double theta)
{
    compute_above(eta);
    compute_below(eta);
    for (int i = 0; i < H; ++i)
    {
        csfs[i] = csfs_above[i] + csfs_below[i];
        T h1 = eta.R(eta.hidden_states[i]), d;
        if (eta.hidden_states[i + 1] < INFINITY)
        {
            T h2 = eta.R(eta.hidden_states[i + 1]);
            d = -exp(-h1) * expm1(h1 - h2);
        }
        else
            d = exp(-h1);
        csfs[i] /= d;
        check_nan(d);
        T tauh = csfs[i].sum();
        check_nan(tauh);
        csfs[i] *= -expm1(-theta * tauh) / tauh;
        T tiny = eta.one * 1e-20;
        csfs[i] = csfs[i].unaryExpr([=](const T x) { if (x < 1e-20) return tiny; if (x < -1e-8) throw std::domain_error("very negative sfs"); return x; });
        tauh = csfs[i].sum();
        csfs[i](0, 0) = 1. - tauh;
        check_nan(csfs[i]);
        if (csfs[i].minCoeff() < 0 or csfs[i].maxCoeff() > 1)
            throw std::runtime_error("csfs is not a probability distribution");
     }
    return csfs;
}

std::map<int, MatrixCache> ConditionedSFSBase::matrix_cache;
MatrixCache& ConditionedSFSBase::cached_matrices(int n)
{
    const MoranEigensystem mei = compute_moran_eigensystem(n);
    if (matrix_cache.count(n) == 0)
    {
        MatrixCache ret;
        std::cout << "computing cached matrices..." << std::endl;
        VectorXq D_subtend_above = VectorXq::LinSpaced(n, 1, n);
        D_subtend_above /= n + 1;

        VectorXq D_subtend_below = Eigen::Array<mpq_class, Eigen::Dynamic, 1>::Ones(n + 1) / 
            Eigen::Array<mpq_class, Eigen::Dynamic, 1>::LinSpaced(n + 1, 2, n + 2);
        D_subtend_below *= 2;

        MatrixXq Wnbj(n, n), P_dist(n + 1, n + 1), P_undist(n + 1, n);
        Wnbj.setZero();
        for (int b = 1; b < n + 1; ++b)
            for (int j = 2; j < n + 2; ++j)
                Wnbj(b - 1, j - 2) = calculate_Wnbj(n + 1, b, j);

        // P_dist(k, b) = probability of state (1, b) when there are k undistinguished lineages remaining
        P_dist.setZero();
        for (int k = 0; k < n + 1; ++k)
            for (int b = 1; b < n - k + 2; ++b)
                P_dist(k, b - 1) = pnkb_dist(n, k, b);

        // P_undist(k, b) = probability of state (0, b + 1) when there are k undistinguished lineages remaining
        P_undist.setZero();
        for (int k = 1; k < n + 1; ++k)
            for (int b = 1; b < n - k + 2; ++b)
                P_undist(k, b - 1) = pnkb_undist(n, k, b);

        ret.X0 = (Wnbj.transpose() * (VectorXq::Ones(n) - D_subtend_above).asDiagonal() * mei.U.bottomRows(n)).template cast<double>();
        ret.X2 = (Wnbj.transpose() * D_subtend_above.asDiagonal() * mei.U.reverse().topRows(n)).template cast<double>();
        below_coeff bc = compute_below_coeffs(n);
        VectorXq lsp = VectorXq::LinSpaced(n + 1, 2, n + 2);
        ret.M0 = (bc.coeffs * lsp.asDiagonal() * (VectorXq::Ones(n + 1) - D_subtend_below).asDiagonal() * P_undist).template cast<double>();
        ret.M1 = (bc.coeffs * lsp.asDiagonal() * D_subtend_below.asDiagonal() * P_dist).template cast<double>();
        // std::ofstream file("matrices.txt", std::ios::out | std::ios::trunc);
        // file << "X0\n" << ret.X0 << "\n\nX2\n" << ret.X2 << "\n\nM0\n" << ret.M0 << "\n\nM1\n" << ret.M1 << std::endl;
        // file.close();
        matrix_cache[n] = ret;
        std::cout << "done" << std::endl;
    }
    return matrix_cache[n];
}

void print_sfs(int n, const std::vector<double> &sfs)
{
    std::vector<double> rsfs(n, 0);
    double x;
    int k = 0;
    for (int i = 0; i < 3; i++)
    {
        printf("%i:\t", i);
        for (int j = 0; j < n; j++)
        {
            x = sfs[k++];
            rsfs[i + j] += x;
            printf("%i:%e ", j, x);
        }
        printf("\n");
    }
    for (int i = 0; i < n; i++)
    {
        printf("%i:%f\n", i, rsfs[i]);
    }
}

template class ConditionedSFS<double>;
template class ConditionedSFS<adouble>;

int csfs_main(int argc, char** argv)
{
    int n = atoi(argv[1]);
    ConditionedSFS<adouble> csfs(n, 50);
    doProgress(true);
    // ConditionedSFS<double> csfs2(0);
    std::vector<std::vector<double> > params = {
        {0.2, 1.0, 2.0, 0.2, 1.0, 2.0, 0.2, 1.0, 2.0},
        {1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0},
        {1.0, 0.1, 0.1, 1.0, 0.1, 0.1, 1.0, 0.1, 0.1}
    };
    std::vector<double> hs;
    for (int i = 0; i < 50; ++i)
        hs.push_back((double)i / 5.0);
    std::vector<std::pair<int, int> > deriv;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 9; ++j)
            deriv.emplace_back(i, j);
    PiecewiseExponentialRateFunction<adouble> eta(params, deriv, hs);
    // params[1][0] += 1e-8;
    // PiecewiseExponentialRateFunction<double> eta2(params, deriv, hs);
    std::vector<Matrix<adouble> > cs = csfs.compute(eta, 4 * 1e-4 * 50);
    for (int h = 0; h < hs.size() - 1; ++h)
    {
        std::cout << h;
        std::cout << cs[h].template cast<double>() << std::endl << std::endl;
        std::cout << cs[h].unaryExpr([](adouble x) { return x.derivatives()(0); }).template cast<double>()
            << std::endl << std::endl;
    }
}
