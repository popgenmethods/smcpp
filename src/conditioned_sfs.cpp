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
        ret.prec = int(log2(mlast.array().abs().maxCoeff().get_d())) + 10;
        if (ret.prec < 53)
            ret.prec = 53;
        ret.coeffs = mlast;
        below_coeffs_memo.emplace(n, ret); 
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
        mpz_class binom1;
        mpz_bin_uiui(binom1.get_mpz_t(), n + 3, m + 3);
        MPQ_CONSTRUCT(ret, l1 * (n + 2 - l1), binom1);
        if (m > 0)
        {
            mpz_class binom2;
            mpz_bin_uiui(binom2.get_mpz_t(), n - l1, m - 1);
            MPQ_CONSTRUCT(r2, (n + 1 - l1) * binom2, m * (m + 1));
            ret *= r2;
        }
        pnkb_dist_memo[key] = ret;
    }
    return pnkb_dist_memo[key];
    /*
     * Numerically unstable for large n
    double ret = l1 * (n + 2 - l1) / (double) binom(n + 3, m + 3);
    if (m > 0)
        ret *= (n + 1 - l1) * (double)binom(n - l1, m - 1) / m / (m + 1);
    */
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
        mpz_class binom1;
        mpz_bin_uiui(binom1.get_mpz_t(), n + 3, m + 3);
        MPQ_CONSTRUCT(ret, (n + 3 - l3) * (n + 2 - l3) * (n + 1 - l3), binom1);
        mpz_class binom2;
        mpz_bin_uiui(binom2.get_mpz_t(), n - l3 - 1, m - 2);
        MPQ_CONSTRUCT(r2, (n - l3) * binom2, (m - 1) * m * (m + 1) * (m + 2));
        if (m == 1)
            ret /= 6;
        else
            ret *= r2;
        pnkb_undist_memo.emplace(key, ret);
    }
    return pnkb_undist_memo[key];
    /*
     * Numerically unstable for large n
    double ret = (n + 3 - l3) * (n + 2 - l3) * (n + 1 - l3) / (double) binom(n + 3, m + 3);
    if (m == 1)
        ret /= 6.0;
    else
        ret *= (n - l3) * (double)binom(n - l3 - 1, m - 2) / (m - 1) / m / (m + 1) / (m + 2);
    return ret;
    */
}

template <typename T>
ConditionedSFS<T>::ConditionedSFS(int n) :
    n(n), 
    mei(compute_moran_eigensystem(n)), mcache(cached_matrices(n)) {}

template <typename T>
template <typename Derived>
Matrix<T> ConditionedSFS<T>::parallel_cwiseProduct_colSum(const MatrixXq &a, const Eigen::MatrixBase<Derived> &b)
{
    Matrix<T> ret(1, a.cols());
    ret.setZero();
#pragma omp parallel for schedule(static)
    for (int j = 0; j < a.cols(); ++j)
        for (int i = 0; i < a.rows(); ++i)
            ret(0, j) += a(i, j).get_d() * b(i, j);
    return ret;
}

template <typename T>
Matrix<T> ConditionedSFS<T>::above0(const Matrix<T> &Ch)
{
    MatrixXq Uinv_mp0 = mei.Uinv.rightCols(n);
    if (n < 20)
        return mcache.X0.template cast<T>().cwiseProduct(Ch.transpose()).colwise().sum() * Uinv_mp0.template cast<T>();
    return parallel_cwiseProduct_colSum(mcache.X0, Ch.transpose()) * Uinv_mp0.template cast<T>();
}

template <typename T>
Matrix<T> ConditionedSFS<T>::above2(const Matrix<T> &Ch)
{
    MatrixXq Uinv_mp2 = mei.Uinv.reverse().leftCols(n);
    if (n < 20)
        return mcache.X2.template cast<T>().cwiseProduct(Ch.colwise().reverse().transpose()).colwise().sum() * 
            Uinv_mp2.template cast<T>();
    return parallel_cwiseProduct_colSum(mcache.X2, Ch.colwise().reverse().transpose()) * Uinv_mp2.template cast<T>();
}

template <typename T>
Matrix<T> ConditionedSFS<T>::below0(const Matrix<mpreal_wrapper<T> > &tjj_below)
{
    // return tjj_below.lazyProduct(mcache.M0).template cast<T>();
    return parallel_matrix_product(tjj_below, mcache.M0).template cast<T>();
}

template <typename T>
Matrix<T> ConditionedSFS<T>::below1(const Matrix<mpreal_wrapper<T> > &tjj_below)
{
    return parallel_matrix_product(tjj_below, mcache.M1).template cast<T>();
    // return tjj_below.lazyProduct(mcache.M1).template cast<T>();
}

template <typename T>
template <typename Derived>
Matrix<mpreal_wrapper<T> > ConditionedSFS<T>::parallel_matrix_product(const Eigen::MatrixBase<Derived> &a, const MatrixXq &b)
{
    Matrix<mpreal_wrapper<T> > ret(a.rows(), b.cols());
    ret.setZero();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < a.rows(); ++i)
    {
        std::vector<std::vector<mpreal_wrapper<T> > > vs(b.cols(), std::vector<mpreal_wrapper<T> >(a.cols()));
        for (int k = 0; k < a.cols(); ++k)
            for (int j = 0; j < b.cols(); ++j)
                vs[j][k] = a(i,k) * b(k,j);
        for (int j = 0; j < b.cols(); ++j)
            ret(i, j) = mpreal_wrapper_type<T>::fsum(vs[j]);
    }
    return ret;
}

template <typename T>
std::vector<Matrix<T> > ConditionedSFS<T>::compute_below(const PiecewiseExponentialRateFunction<T> &eta)
{
    mpfr::mpreal::set_default_prec(mcache.prec);
    PROGRESS("compute below");
    Matrix<mpreal_wrapper<T> > ts_integrals(eta.K, n + 1); 
    long wprec = (long)mcache.prec;
    double log2d, h1, h2;
    for (int h = 0; h < eta.hidden_states.size() - 1; ++h)
    {
        h1 = toDouble(eta.R(eta.hidden_states[h]));
        log2d = h1 / log(2);
        if (eta.hidden_states[h + 1] < INFINITY)
        {
            h2 = toDouble(eta.R(eta.hidden_states[h + 1]));
            log2d = -h1 / log(2) + log2(-expm1(h1 - h2));
        }
        else
            log2d = -h1 / log(2);
        wprec = std::max((long)std::ceil(-log2d), wprec);
    }
#pragma omp parallel for
    for (int m = 0; m < eta.K; ++m)
        eta.tjj_double_integral_below(n, wprec, m, ts_integrals);
    size_t H = eta.hidden_states.size() - 1;
    Matrix<mpreal_wrapper<T> > tjj_below(H, n + 1);
    Matrix<mpreal_wrapper<T> > last = ts_integrals.topRows(eta.hs_indices[0]).colwise().sum(), next;
    for (int h = 1; h < H + 1; ++h)
    {
        next = ts_integrals.topRows(eta.hs_indices[h]).colwise().sum();
        tjj_below.row(h - 1) = next - last;
        last = next;
    }

    // std::cout << "tjj_below:\n" << tjj_below.template cast<T>().template cast<double>() << std::endl << std::endl;

    PROGRESS("matrix products below");
    Matrix<T> M0_below = below0(tjj_below);
    Matrix<T> M1_below = below1(tjj_below);
    std::vector<Matrix<T> > ret(H, Matrix<T>::Zero(3, n + 1));
    PROGRESS("mpfr sfs below");
    for (int h = 0; h < H; ++h) 
    {
        ret[h].block(0, 1, 1, n) = M0_below.row(h);
        ret[h].block(1, 0, 1, n + 1) = M1_below.row(h);
    }
    PROGRESS_DONE();
    return ret;
}

template <typename T>
std::vector<Matrix<T> > ConditionedSFS<T>::compute_above(
    const PiecewiseExponentialRateFunction<T> &eta
    )
{
    PROGRESS("compute above");
    int H = eta.hidden_states.size() - 1;
    PROGRESS("tjj double integral");
    std::vector<Matrix<T> > C(H, Matrix<T>::Zero(n + 1, n)), ret(H, Matrix<T>::Zero(3, n + 1));
#pragma omp parallel for
    for (int j = 2; j < n + 3; ++j)
        eta.tjj_double_integral_above(n, j, C);
    Matrix<T> tmp;
    PROGRESS("matrix products");

    // for (int h = 0; h < H; ++h)
        // std::cout << "tjj_above (" << h << "):\n" << C[h].template cast<T>().template cast<double>() << std::endl << std::endl;

    for (int h = 0; h < H; ++h)
    {
        ret[h].block(0, 1, 1, n) += above0(C[h]);
        ret[h].block(2, 0, 1, n) += above2(C[h]);
    }
    PROGRESS_DONE();
    return ret;
}

template <typename T>
std::vector<Matrix<T> > ConditionedSFS<T>::compute(const PiecewiseExponentialRateFunction<T> &eta, double theta)
{
    std::vector<Matrix<T> > above = compute_above(eta), below = compute_below(eta);
    std::vector<Matrix<T> > ret(above.size());
    for (int i = 0; i < above.size(); ++i)
    {
        ret[i] = above[i] + below[i];
        T h1 = eta.R(eta.hidden_states[i]), d;
        if (eta.hidden_states[i + 1] < INFINITY)
        {
            T h2 = eta.R(eta.hidden_states[i + 1]);
            d = -exp(-h1) * expm1(h1 - h2);
        }
        else
            d = exp(-h1);
        ret[i] /= d;
        T tauh = ret[i].sum();
        ret[i] *= -expm1(-theta * tauh) / tauh;
        ret[i](0, 0) = exp(-theta * tauh);
        T tiny = (ret[i](0, 0) - ret[i](0, 0)) + 1e-20;
        ret[i] = ret[i].unaryExpr([=](T x) { if (x < 1e-20) return tiny; if (x < -1e-8) throw std::domain_error("very negative sfs"); return x; });
        check_nan(ret[i]);
     }
    return ret;
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

        ret.X0 = Wnbj.transpose() * (VectorXq::Ones(n) - D_subtend_above).asDiagonal() * mei.U.bottomRows(n);
        ret.X2 = Wnbj.transpose() * D_subtend_above.asDiagonal() * mei.U.reverse().topRows(n);

        below_coeff bc = compute_below_coeffs(n);
        VectorXq lsp = VectorXq::LinSpaced(n + 1, 2, n + 2);
        ret.M0 = bc.coeffs * lsp.asDiagonal() * (VectorXq::Ones(n + 1) - D_subtend_below).asDiagonal() * P_undist;
        ret.M1 = bc.coeffs * lsp.asDiagonal() * D_subtend_below.asDiagonal() * P_dist;
        if (n > 0)
        {
            mpq_class m1 = ret.M0.array().abs().maxCoeff();
            mpq_class m2 = ret.M1.array().abs().maxCoeff();
            ret.prec = std::max(53, (int)mpfr::mpreal(std::max(m1, m2).get_mpq_t()).get_exp() + 10);
        }
        else
            ret.prec = 53;
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
