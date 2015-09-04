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
                mnew.col(k - 2) = mlast.col(k - 2) * mpq_class((nn + 1) * (nn - 2), denom);
                mnew.col(k - 2) -= mnew.col(k - 1) * mpq_class((k + 2) * (k - 1), denom);
            }
            mlast = mnew;
        }
        ret.prec = int(log2(mlast.array().abs().maxCoeff().get_d())) + 10;
        if (ret.prec < 53)
            ret.prec = 53;
        ret.coeffs = mlast;
        below_coeffs_memo[n] = ret;
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
            return mpq_class(30 * (n - 2 * b), (n + 1) * (n + 2));
        default:
            std::array<int, 3> key = {n, b, j};
            if (_Wnbj_memo.count(key) == 0)
            {
                int jj = j - 2;
                mpq_class ret = calculate_Wnbj(n, b, jj) * mpq_class(-(1 + jj) * (3 + 2 * jj) * (n - jj), jj * (2 * jj - 1) * (n + jj + 1));
                ret += calculate_Wnbj(n, b, jj + 1) * mpq_class((3 + 2 * jj) * (n - 2 * b), jj * (n + jj + 1));
                _Wnbj_memo[key] = ret;
            }
            return _Wnbj_memo[key];
    }
}

std::map<std::array<int, 2>, long> _binom_memo;
long binom(int n, int k)
{
    assert(k >= 0);
    if (k == 0 or n == k)
        return 1;
    std::array<int, 2> key = {n, k};
    if (_binom_memo.count(key) == 0) 
        _binom_memo[key] = binom(n - 1, k - 1) + binom(n - 1, k);
    return _binom_memo[key];
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
        mpq_class ret(l1 * (n + 2 - l1), binom1);
        if (m > 0)
        {
            mpz_class binom2;
            mpz_bin_uiui(binom2.get_mpz_t(), n - l1, m - 1);
            mpq_class r2((n + 1 - l1) * binom2, m * (m + 1));
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
        mpq_class ret((n + 3 - l3) * (n + 2 - l3) * (n + 1 - l3), binom1);
        if (m == 1)
            ret /= 6;
        else
        {
            mpz_class binom2;
            mpz_bin_uiui(binom2.get_mpz_t(), n - l3 - 1, m - 2);
            mpq_class r2((n - l3) * binom2, (m - 1) * m * (m + 1) * (m + 2));
            ret *= r2;
        }
        pnkb_undist_memo[key] = ret;
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
ConditionedSFS<T>::ConditionedSFS(int n, int num_threads) : 
    n(n), num_threads(num_threads),
    mei(compute_moran_eigensystem(n)), mcache(cached_matrices(n)),
    csfs(3, n + 1), csfs_above(3, n + 1), csfs_below(3, n + 1), tp(num_threads) { Eigen::setNbThreads(num_threads); }

/*
template <typename T>
Matrix<T> ConditionedSFS<T>::compute_etnk_below_mat(const Matrix<mpreal_wrapper<T> > &etjj)
{
    Matrix<T> ret(etjj.rows(), etjj.cols());
    ret.setZero();
    // This basically expresses:
    // ret = T * bc where T(i, j) = E(Tjj | H[i])
    for (int i = 0; i < ret.rows(); ++i)
        for (int j = 0; j < ret.cols(); ++j)
            {
                mpreal_wrapper<T> tmp(0.0);
                for (int k = 0; k < ret.cols(); ++k)
                    tmp += etjj(i, k) * bc.coeffs(k, j).get_mpq_t();
                ret(i, j) = mpreal_wrapper_convertBack<T>((j + 2) * tmp);
                if (ret(i, j) < -1e-4)
                {
                    std::cout << "(" << i << "," << j << "):\n" << ret.template cast<double>() << std::endl;
                    throw std::domain_error("highly negative etnk entry");
                }
                ret(i, j) = dmax(ret(i, j), 1e-20);
            }
    return ret;
}
*/

template <typename T>
template <typename Derived>
Matrix<T> ConditionedSFS<T>::parallel_cwiseProduct_colSum(const MatrixXq &a, const Eigen::MatrixBase<Derived> &b)
{
    Matrix<T> ret(1, a.cols());
    ret.setZero();
#pragma omp parallel for
    for (int j = 0; j < a.cols(); ++j)
        for (int i = 0; i < a.rows(); ++i)
            ret(0, j) += a(i, j).get_d() * b(i, j);
    /*std::vector<std::future<void> > res;
    int bsize = std::ceil((double)a.cols() / num_threads), start, end;
    for (int t = 0; t < num_threads; ++t)
    {
        start = t * bsize;
        end = std::min((t + 1) * bsize, (int)a.rows());
        res.emplace_back(tp.enqueue([start, end, &a, &b, &ret]
            {
                for (int j = start; j < end; ++j)
                    for (int i = 0; i < a.rows(); ++i)
                        ret(0, j) += a(i, j).get_d() * b(i, j);
            }));
    }
    for (int t = 0; t < num_threads; ++t)
        res[t].wait();
        */
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
    if (n < 20)
        return (tjj_below * mcache.M0.template cast<mpreal_wrapper<T> >()).template cast<T>();
    return parallel_matrix_product(tjj_below, mcache.M0).template cast<T>();
}

template <typename T>
Matrix<T> ConditionedSFS<T>::below1(const Matrix<mpreal_wrapper<T> > &tjj_below)
{
    if (n < 20)
        return (tjj_below * mcache.M1.template cast<mpreal_wrapper<T> >()).template cast<T>();
    return parallel_matrix_product(tjj_below, mcache.M1).template cast<T>();
    // return tjj_below.lazyProduct(mcache.M1).template cast<T>();
}

template <typename T>
template <typename Derived>
Matrix<mpreal_wrapper<T> > ConditionedSFS<T>::parallel_matrix_product(const Eigen::MatrixBase<Derived> &a, const MatrixXq &b)
{
    Matrix<mpreal_wrapper<T> > ret(a.rows(), b.cols());
    ret.setZero();
#pragma omp parallel for
    for (int i = 0; i < a.rows(); ++i)
        for (int k = 0; k < a.cols(); ++k)
            for (int j = 0; j < b.cols(); ++j)
                ret(i, j) += a(i, k) * b(k, j);
    /*
    std::vector<std::future<void> > res;
    int bsize = std::ceil((double)a.rows() / num_threads), start, end;
    for (int t = 0; t < num_threads; ++t)
    {
        start = t * bsize;
        end = std::min((t + 1) * bsize, (int)a.rows());
        res.emplace_back(tp.enqueue([start, end, &a, &b, &ret]
            {
                for (int i = start; i < end; ++i)
                    for (int k = 0; k < a.cols(); ++k)
                        for (int j = 0; j < b.cols(); ++j)
                            ret(i, j) += a(i, k) * b(k, j);
            }));
    }
    for (int t = 0; t < num_threads; ++t)
        res[t].wait();
        */
    return ret;
}

template <typename T>
std::vector<Matrix<T> > ConditionedSFS<T>::compute_below(const PiecewiseExponentialRateFunction<T> &eta)
{
    mpfr::mpreal::set_default_prec(mcache.prec);
    PROGRESS("compute below");
    Matrix<mpreal_wrapper<T> > ts_integrals(eta.K, n + 1); 
    std::vector<std::future<void> > res;
#pragma omp parallel for
    for (int m = 0; m < eta.K; ++m)
        eta.tjj_double_integral_below(n, mcache.prec, m, ts_integrals);
    /*
    for (int m = 0; m < eta.K; ++m)
        res.emplace_back(tp.enqueue([this, eta, m, &ts_integrals] { eta.tjj_double_integral_below(n, mcache.prec, m, ts_integrals); }));
    for (int m = 0; m < eta.K; ++m)
        res[m].wait();
        */
    size_t H = eta.hidden_states.size() - 1;
    Matrix<mpreal_wrapper<T> > tjj_below(H, n + 1);
    Matrix<mpreal_wrapper<T> > last = ts_integrals.topRows(eta.hs_indices[0]).colwise().sum(), next;
    for (int h = 1; h < H + 1; ++h)
    {
        next = ts_integrals.topRows(eta.hs_indices[h]).colwise().sum();
        tjj_below.row(h - 1) = next - last;
        last = next;
    }
    PROGRESS("matrix products below");
    /*
    std::vector<std::future<Matrix<T> > > res2;
    res2.emplace_back(tp.enqueue([&tjj_below, this] { return below0(tjj_below); }));
    res2.emplace_back(tp.enqueue([&tjj_below, this] { return below1(tjj_below); }));
    */
    Matrix<T> M0_below = below0(tjj_below);
    Matrix<T> M1_below = below1(tjj_below);
    std::vector<Matrix<T> > ret(H, Matrix<T>::Zero(3, n + 1));
    PROGRESS("mpfr sfs below");
    T h1(0.0), h2(0.0);
    for (int h = 0; h < H; ++h) 
    {
        ret[h].block(0, 1, 1, n) = M0_below.row(h);
        ret[h].block(1, 0, 1, n + 1) = M1_below.row(h);
        h1 = exp(-(*(eta.getR()))(eta.hidden_states[h]));
        if (eta.hidden_states[h + 1] == INFINITY)
            h2 *= 0.0;
        else
            h2 = exp(-(*(eta.getR()))(eta.hidden_states[h + 1]));
        ret[h] /= h1 - h2;
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
    std::vector<std::future<void> > res;
    PROGRESS("tjj double integral");
    std::vector<Matrix<T> > C(H, Matrix<T>::Zero(n + 1, n)), ret(H, Matrix<T>::Zero(3, n + 1));
#pragma omp parallel for
    for (int j = 2; j < n + 3; ++j)
        eta.tjj_double_integral_above(n, j, C);
    /*
    for (int j = 2; j < n + 3; ++j)
        res.emplace_back(tp.enqueue([eta, this, j, &C] { eta.tjj_double_integral_above(n, j, C); }));
    for (int j = 2; j < n + 3; ++j)
        res[j - 2].wait();
        */
    Matrix<T> tmp;
    PROGRESS("matrix products");
    /*
    res.clear();
    for (int h = 0; h < H; ++h)
    {
        res.emplace_back(tp.enqueue([this, &C, h] { return this->above0(C[h]); }));
        res2.emplace_back(tp.enqueue([this, &C, h] { return this->above2(C[h]); }));
    }
    */
    for (int h = 0; h < H; ++h)
    {
        ret[h].block(0, 1, 1, n) += above0(C[h]);
        ret[h].block(2, 0, 1, n) += above2(C[h]);
        T h1(0.0), h2(0.0);
        h1 = exp(-(*(eta.getR()))(eta.hidden_states[h]));
        if (eta.hidden_states[h + 1] < INFINITY)
            h2 = exp(-(*(eta.getR()))(eta.hidden_states[h + 1]));
        ret[h] /= h1 - h2;
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
        ret[i] = ret[i].unaryExpr([](T x) { if (x < 1e-20) return T(1e-20); return x; });
        T tauh = ret[i].sum();
        ret[i] *= -expm1(-theta * tauh) / tauh;
        ret[i](0, 0) = exp(-theta * tauh);
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
        ret.M0 = bc.coeffs * lsp.asDiagonal() * (VectorXq::Ones(n) - D_subtend_below).asDiagonal() * P_undist;
        ret.M1 = bc.coeffs * lsp.asDiagonal() * D_subtend_below.asDiagonal() * P_dist;
        ret.prec = std::max(53, int(log2(std::max(
                        ret.M0.array().abs().maxCoeff().get_d(),
                        ret.M1.array().abs().maxCoeff().get_d()))) + 10);
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

void store_sfs_results(const Matrix<double> &csfs, double* outsfs)
{
    int n = csfs.cols();
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> _outsfs(outsfs, 3, n);
    _outsfs = csfs.cast<double>();
}

void store_sfs_results(const Matrix<adouble> &csfs, double* outsfs, double* outjac)
{
    store_sfs_results(csfs.cast<double>(), outsfs);
    int n = csfs.cols();
    int num_derivatives = csfs(0,0).derivatives().rows();
    Eigen::VectorXd d;
    int m = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < n; ++j)
        {
            d = csfs(i, j).derivatives();
            assert(d.rows() == num_derivatives);
            for (int k = 0; k < num_derivatives; ++k)
                outjac[m++] = d(k);
        }
}

void cython_calculate_sfs(const std::vector<std::vector<double>> params,
        int n, double tau1, double tau2, int num_threads, double theta, 
        double* outsfs)
{
    std::vector<double> hidden_states = {tau1, tau2};
    PiecewiseExponentialRateFunction<double> eta(params, hidden_states);
    ConditionedSFS<double> csfs(n, num_threads);
    Matrix<double> out = csfs.compute(eta, theta)[0];
    store_sfs_results(out, outsfs);
}

void cython_calculate_sfs_jac(const std::vector<std::vector<double>> params,
        int n, double tau1, double tau2, int num_threads, double theta, 
        double* outsfs, double* outjac)
{
    PiecewiseExponentialRateFunction<adouble> eta(params, {tau1, tau2});
    ConditionedSFS<adouble> csfs(n, num_threads);
    Matrix<adouble> out = csfs.compute(eta, theta)[0];
    store_sfs_results(out, outsfs, outjac);
}

template class ConditionedSFS<double>;
template class ConditionedSFS<adouble>;
