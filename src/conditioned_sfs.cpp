#include "conditioned_sfs.h"

#if 0
#undef PROGRESS
#undef PROGRESS_DONE
#define PROGRESS(x) progress_mtx.lock(); std::cout << x << "... " << std::flush; progress_mtx.unlock();
#define PROGRESS_DONE() progress_mtx.lock(); std::cout << "done." << std::endl << std::flush; progress_mtx.unlock();
#endif

std::mutex progress_mtx;

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
        ret.prec = int(log2(mlast.array().abs().maxCoeff().get_d())) + 20;
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
    mei(compute_moran_eigensystem(n)),
    bc(compute_below_coeffs(n)),
	Wnbj(cached_matrices(n)[0]),
    P_dist(cached_matrices(n)[1]),
    P_undist(cached_matrices(n)[2]), 
    X0(cached_matrices(n)[3]), X2(cached_matrices(n)[4]),
    D_subtend_above(cached_matrices(n)[5]), 
    D_subtend_below(cached_matrices(n)[6]),
    csfs(3, n + 1), csfs_above(3, n + 1), csfs_below(3, n + 1)
{}

template <typename T>
Matrix<T> ConditionedSFS<T>::compute_etnk_below_mat(const Matrix<mpreal_wrapper<T> > &etjj)
{
    Matrix<T> ret(etjj.rows(), etjj.cols());
    mpreal_wrapper<T> tmp;
    for (int i = 0; i < ret.rows(); ++i)
        for (int j = 0; j < ret.cols(); ++j)
            {
                tmp -= tmp;
                for (int k = 0; k < ret.cols(); ++k)
                    tmp += etjj(i, k) * bc.coeffs(k, j).get_mpq_t();
                ret(i, j) = mpreal_wrapper_convertBack<T>((j + 2) * tmp);
                if (ret(i, j) < -1e-8)
                    throw std::domain_error("highly negative etnk entry");
                ret(i, j) = dmax(ret(i, j), 1e-20);
            }
    return ret;
}

void print_derivatives(const Matrix<adouble> &x) { std::cout << x(0, 1).derivatives() << std::endl; }

template <typename T>
std::vector<Matrix<T> > ConditionedSFS<T>::compute_below(
    const PiecewiseExponentialRateFunction<T> &eta
    )
{
    mpfr::mpreal::set_default_prec(bc.prec);
    PROGRESS("mpfr double integration below");
    Matrix<mpreal_wrapper<T> > tjj_below = eta.tjj_double_integral_below(n, bc.prec);
    PROGRESS("mpfr etnk");
    Matrix<T> etnk_below = compute_etnk_below_mat(tjj_below);
    int H = etnk_below.rows();
    std::vector<Matrix<T> > ret(H, Matrix<T>::Zero(3, n + 1));
    Vector<T> ones = Vector<T>::Ones(n + 1);
    PROGRESS("mpfr sfs below");
    for (int i = 0; i < H; ++i) 
    {
        ret[i].block(0, 1, 1, n) = etnk_below.row(i).transpose().
            cwiseProduct(ones - D_subtend_below.template cast<T>()).transpose() * P_undist.template cast<double>();
        ret[i].block(1, 0, 1, n + 1) = etnk_below.row(i).transpose().cwiseProduct(D_subtend_below.template cast<T>()).
            transpose() * P_dist.template cast<double>();
    }
    PROGRESS("compute below done");
    return ret;
}

template <typename T>
std::vector<Matrix<T> > ConditionedSFS<T>::compute_above(
    const PiecewiseExponentialRateFunction<T> &eta
    )
{
    mpfr::mpreal::set_default_prec(53);
    PROGRESS("mpfr double integration");
    MoranEigensystem mei = compute_moran_eigensystem(n);
    int H = eta.hidden_states.size() - 1;
    std::vector<Matrix<mpreal_wrapper<T> > > C(H, Matrix<mpreal_wrapper<T> >::Zero(n + 1, n));
    ThreadPool tp(8);
    std::vector<std::future<Matrix<mpreal_wrapper<T> > > > results;
    int nn = n;
    std::vector<double> hidden_states = eta.hidden_states;
    for (int h = 0; h < H; ++h)
    {
        PiecewiseExponentialRateFunction<T> e2(eta.params, eta.derivatives, {hidden_states[h], hidden_states[h + 1]});
        results.emplace_back(tp.enqueue([e2, nn] { return e2.template tjj_all_above(nn); }));
        // std::cout << h << " " << std::flush;
        // C[h] = e2.template tjj_all_above(nn);
    }
    for (int h = 0; h < H; ++h)
        C[h] = results[h].get();
    MatrixXq Uinv_mp0 = mei.Uinv.rightCols(n);
    MatrixXq Uinv_mp2 = mei.Uinv.reverse().leftCols(n);
    Matrix<mpreal_wrapper<T> > T_subtend(1, n);
    /*
    Matrix<double> sfs = Wnbj.template cast<double>() * C[0].row(0).transpose().template cast<double>();
    std::cout << "sfs " << sfs.transpose() << std::endl;
    */
    std::vector<Matrix<T> > ret(H, Matrix<T>::Zero(3, n + 1));
    for (int h = 0; h < H; ++h) 
    {
        T_subtend = ((X0.template cast<mpreal_wrapper<T> >().cwiseProduct(C[h].transpose()).colwise().sum()) * 
                Uinv_mp0.template cast<mpreal_wrapper<T> >());
        ret[h].block(0, 1, 1, n) += T_subtend.template cast<T>();
        T_subtend = ((X2.template cast<mpreal_wrapper<T> >().cwiseProduct(C[h].colwise().reverse().transpose()).colwise().sum()) * 
                Uinv_mp2.template cast<mpreal_wrapper<T> >());
        ret[h].block(2, 0, 1, n) += T_subtend.template cast<T>();
    }
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
        T tauh = ret[i].sum();
        ret[i] *= -expm1(-theta * tauh) / tauh;
        ret[i](0, 0) = exp(-theta * tauh);
    }
    return ret;
}

std::map<int, std::array<MatrixXq, 7> > ConditionedSFSBase::matrix_cache;
std::array<MatrixXq, 7>& ConditionedSFSBase::cached_matrices(int n)
{
    const MoranEigensystem mei = compute_moran_eigensystem(n);
    if (matrix_cache.count(n) == 0)
    {
        std::cout << "computing cached matrices..." << std::endl;
        VectorXq _D_subtend_above = VectorXq::LinSpaced(n, 1, n);
        _D_subtend_above /= n + 1;

        VectorXq _D_subtend_below = Eigen::Array<mpq_class, Eigen::Dynamic, 1>::Ones(n + 1) / 
            Eigen::Array<mpq_class, Eigen::Dynamic, 1>::LinSpaced(n + 1, 2, n + 2);
        _D_subtend_below *= 2;

        MatrixXq _Wnbj(n, n), _P_dist(n + 1, n + 1), _P_undist(n + 1, n), _X0, _X2;
        _Wnbj.setZero();
        for (int b = 1; b < n + 1; ++b)
            for (int j = 2; j < n + 2; ++j)
                _Wnbj(b - 1, j - 2) = calculate_Wnbj(n + 1, b, j);

        // P_dist(k, b) = probability of state (1, b) when there are k undistinguished lineages remaining
        _P_dist.setZero();
        for (int k = 0; k < n + 1; ++k)
            for (int b = 1; b < n - k + 2; ++b)
                _P_dist(k, b - 1) = pnkb_dist(n, k, b);

        // P_undist(k, b) = probability of state (0, b + 1) when there are k undistinguished lineages remaining
        _P_undist.setZero();
        for (int k = 1; k < n + 1; ++k)
            for (int b = 1; b < n - k + 2; ++b)
                _P_undist(k, b - 1) = pnkb_undist(n, k, b);

        _X0 = _Wnbj.transpose() * (VectorXq::Ones(n) - _D_subtend_above).asDiagonal() * mei.U.bottomRows(n);
        _X2 =  _Wnbj.transpose() * _D_subtend_above.asDiagonal() * mei.U.reverse().topRows(n);
        matrix_cache[n] = {_Wnbj, _P_dist, _P_undist, _X0, _X2, _D_subtend_above, _D_subtend_below};
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
