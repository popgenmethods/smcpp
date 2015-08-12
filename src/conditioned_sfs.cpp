#include "conditioned_sfs.h"

#if 0
#undef PROGRESS
#undef PROGRESS_DONE
#define PROGRESS(x) progress_mtx.lock(); std::cout << x << "... " << std::flush; progress_mtx.unlock();
#define PROGRESS_DONE() progress_mtx.lock(); std::cout << "done." << std::endl << std::flush; progress_mtx.unlock();
#else
#define PROGRESS(x)
#define PROGRESS_DONE()
#endif

std::mutex progress_mtx;

std::map<int, below_coeff> below_coeffs_memo;
below_coeff compute_below_coeffs(int n)
{
    if (below_coeffs_memo.count(n) == 0)
    {
        PROGRESS("Computing below_coeffs");
        below_coeff ret;
        std::valarray<mpq_class> a(0_mpq, n + 1);
        std::vector<std::valarray<mpq_class>> mlast;
        for (int nn = 2; nn < n + 3; ++nn)
        {
            std::vector<std::valarray<mpq_class>> mnew(nn - 1);
            mnew[nn - 2] = a;
            mnew[nn - 2][nn - 2] = 1_mpq;
            for (int k = nn - 1; k > 1; --k)
            {
                mpz_class denom = (nn + 1) * (nn - 2) - (k + 1) * (k - 2);
                mnew[k - 2] = mlast[k - 2] * mpq_class((nn + 1) * (nn - 2), denom);
                mnew[k - 2] -= mnew[k - 1] * mpq_class((k + 2) * (k - 1), denom);
                /*
                m[{nn - 2, k - 2}] = m[{nn - 3, k - 2}] * mpq_class((nn + 1) * (nn - 2), denom);
                m[{nn - 2, k - 2}] -= m[{nn - 2, k - 1}] * mpq_class((k + 1) * (k - 2), denom);
                */
            }
            mlast = mnew;
        }
        double p = 1;
        for (int k = 2; k < n + 3; ++k)
            p = std::accumulate(std::begin(mlast[k - 2]), std::end(mlast[k - 2]), p,
                    [] (double pp, mpq_class &x) 
                    {
                        mpfr::mpreal r(x.get_mpq_t());
                        return std::max(pp, log10(abs(r)).toDouble());
                    });
        ret.prec = mpfr::digits2bits(p + 20);
        if (ret.prec < 53)
            ret.prec = 53;
        ret.coeffs = mlast;
        below_coeffs_memo[n] = ret;
        PROGRESS_DONE();
    }
    return below_coeffs_memo[n];
}

std::map<std::array<int, 3>, double> _Wnbj_memo;
double calculate_Wnbj(int n, int b, int j)
{
    switch (j)
    {
        case 2:
            return (double)6 / (n + 1);
        case 3:
            return (double)30 * (n - 2 * b) / (n + 1) / (n + 2);
        default:
            std::array<int, 3> key = {n, b, j};
            if (_Wnbj_memo.count(key) == 0)
            {
                int jj = j - 2;
                double ret = calculate_Wnbj(n, b, jj) * -(1 + jj) * (3 + 2 * jj) * (n - jj) / jj / (2 * jj - 1) / (n + jj + 1);
                ret += calculate_Wnbj(n, b, jj + 1) * (3 + 2 * jj) * (n - 2 * b) / jj / (n + jj + 1);
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

std::map<std::array<int, 3>, double> pnkb_dist_memo;
double pnkb_dist(int n, int m, int l1)
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
        pnkb_dist_memo[key] = ret.get_d();
    }
    return pnkb_dist_memo[key];
    /*
     * Numerically unstable for large n
    double ret = l1 * (n + 2 - l1) / (double) binom(n + 3, m + 3);
    if (m > 0)
        ret *= (n + 1 - l1) * (double)binom(n - l1, m - 1) / m / (m + 1);
    */
}

std::map<std::array<int, 3>, double> pnkb_undist_memo;
double pnkb_undist(int n, int m, int l3)
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
        pnkb_undist_memo[key] = ret.get_d();
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
ConditionedSFS<T>::ConditionedSFS(int n, MatrixInterpolator moran_interp) :
    n(n), moran_interp(moran_interp), 
    D_subtend_above(n), D_subtend_below(n + 1),
	Wnbj(cached_matrices(n)[0]), P_dist(cached_matrices(n)[1]),
    P_undist(cached_matrices(n)[2]), csfs(3, n + 1), 
    csfs_above(3, n + 1), csfs_below(3, n + 1), bc(compute_below_coeffs(n))
{
    fill_matrices();
    mpfr::mpreal::set_default_prec(bc.prec);
}

template <typename T>
double ConditionedSFS<T>::rand_exp(void)
{
    return std::exponential_distribution<double>{1.0}(gen);
}

template <typename T>
double ConditionedSFS<T>::unif(void)
{
    return std::uniform_real_distribution<double>{0.0, 1.0}(gen);
}

template <typename T>
T ConditionedSFS<T>::exp1_conditional(T a, T b)
{
    // If X ~ Exp(1),
    // P(X < x | a <= X <= b) = (e^-a - e^-x) / (e^-a - e^-b)
    // so P^-1(y) = -log(e^-a - (e^-a - e^-b) * y)
    //            = -log(e^-a(1 - (1 - e^-(b-a)) * y)
    //            = a - log(1 - (1 - e^-(b-a)) * y)
    //            = a - log(1 + expm1(-(b-a)) * y)
    if (std::isinf(toDouble(b)))
        return a - log1p(-unif());
    else
        return a - log1p(expm1(-(b - a)) * unif());
}

template <typename T>
Vector<T> ConditionedSFS<T>::compute_etnk_below(const std::vector<mpreal_wrapper<T>> &etjj)
{
    Vector<T> ret = Vector<T>::Zero(n + 1);
    ret.fill(0);
    std::valarray<mpreal_wrapper<T>> v(n + 1);
    for (int i = 0; i < n + 1; ++i)
        v[i] = etjj[i];
    mpreal_wrapper<T> tmp;
    for (int i = 0; i < n + 1; ++i)
    {
        tmp -= tmp;
        for (int j = 0; j < n + 1; ++j)
            tmp += v[j] * bc.coeffs[i][j].get_mpq_t();
        ret(i) = mpreal_wrapper_convertBack<T>((i + 2) * tmp);
    }
    return ret;
}

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
                    tmp += etjj(i, k) * bc.coeffs[j][k].get_mpq_t();
                ret(i, j) = mpreal_wrapper_convertBack<T>((j + 2) * tmp);
            }
    return ret;
}

void check_for_nans(Vector<double> x) 
{
    for (int i = 0; i < x.rows(); ++i)
        if (std::isnan(x(i)))
            throw std::domain_error("got nans in x");
}

void check_for_nans(Vector<adouble> x) 
{
    Vector<double> vd = x.template cast<double>();
    check_for_nans(vd);
    for (int i = 0; i < x.rows(); ++i)
        check_for_nans(x(i).derivatives());
}


template <typename T>
std::vector<Matrix<T> > ConditionedSFS<T>::compute_below(
    const PiecewiseExponentialRateFunction<T> &eta, 
    const std::vector<double> hidden_states)
{
    PROGRESS("mpfr double integration");
    Matrix<mpreal_wrapper<T> > mpfr_tjj_below = eta.mpfr_tjj_double_integral(n, hidden_states, bc.prec);
    PROGRESS("mpfr etnk");
    Matrix<T> etnk_below = compute_etnk_below_mat(mpfr_tjj_below);
    std::vector<Matrix<T> > ret(hidden_states.size() - 1, Matrix<T>::Zero(3, n + 1));
    Vector<T> ones = Vector<T>::Ones(n + 1);
    PROGRESS("mpfr sfs below");
    for (int i = 0; i < hidden_states.size() - 1; ++i) 
    {
        ret[i].block(0, 1, 1, n) = etnk_below.row(i).transpose().
            cwiseProduct(ones - D_subtend_below).transpose() * P_undist;
        ret[i].block(1, 0, 1, n + 1) = etnk_below.row(i).transpose().cwiseProduct(D_subtend_below).transpose() * P_dist;
    }
    PROGRESS("compute below done");
    return ret;
}

template <typename T>
void ConditionedSFS<T>::compute(const PiecewiseExponentialRateFunction<T> &eta, int num_samples, T t1, T t2)
{
    // feenableexcept(FE_INVALID | FE_OVERFLOW);
    PROGRESS("compute above");
    auto Rinv = eta.getRinv();
    T tau;
    // There are n + 2 (undistinguished + distinguished) at time 0.
    // Above tau there are between 2 and n + 1.
        // For some reason it does not like initializing these as AdVectors. So use
        // the matrix class instead.
    Vector<T> etjj_above(n), etjj_below(n + 1), sfs_tau(n, 1);
    Vector<T> tjj_above(n), tjj_below(n + 1);
    std::vector<mpreal_wrapper<T>> mpfr_tjj_below(n + 1);
    Vector<T> gauss_tjj_above(n), gauss_tjj_below(n + 1);
    Vector<T> etnk_below(n + 1), etnk_above(n);
    Matrix<T> eM(n + 1, n + 1);
    int m;
    T y, Rt;
    Matrix<T> tmpmat;

    // Mixing constants with adoubles causes problems because the library
    // doesn't know how to allocate the VectorXd of derivatives().
    // Here, we do it by hand.
    T zero = eta.zero;
    csfs.fill(zero);
    csfs_above.fill(zero);
    csfs_below.fill(zero);
    std::vector<T> ys(num_samples);
    std::generate(ys.begin(), ys.end(), [&](){return exp1_conditional(t1, t2);});
    std::sort(ys.begin(), ys.end());
    std::vector<int> eis(num_samples);
    std::vector<T> taus = (*Rinv)(ys);
    for (m = 0; m < num_samples; ++m)
    {
        tau = taus[m];
        y = ys[m];
        // mpfr_tjj_below[0] = mpreal_wrapper_convert<T>(tau);
        for (int j = 2; j < n + 2; ++j)
        {
            // unsigned long int lrate = (j + 1) * j / 2 - 1;
            // mpfr_tjj_below[j - 1] = eta.mpfr_tjj_integral(lrate, zero, tau, zero, bc.prec);
            double rate = j * (j - 1) / 2;
            tjj_above(j - 2) = eta.tjj_integral(rate, tau, INFINITY, y);
        }
        // Compute sfs above using polanski-kimmel + matrix exponential
        // Get the correct linear-interpolated matrix exponential
        eM = moran_interp.interpolate<T>(y);
        // Wnbj is the Polanski-Kimmel matrix of coefficients W_(n + 1, b, j)
        sfs_tau = Wnbj * tjj_above; // n-vector; sfs_tau[b] = prob. lineage subtends b + 1
        // If a = 0 (neither distinguished lineage is derived) then immediately after
        // tau, it's sum_b p(|B|=b at tau) * ({1,2} not a subset of B) * p(|B| -> (0, k))
        // Recall that the rate matrix for the a = 0 case is the reverse of the a = 2 case,
        // which is what gets passed in.
        tmpmat = eM.reverse().bottomRightCorner(n, n).transpose() * (Vector<T>::Ones(n) - D_subtend_above).asDiagonal() / num_samples;
		csfs_above.block(0, 1, 1, n) += (tmpmat * sfs_tau).transpose();
        // If a = 2 (both distinguished lineages derived) then immediately after
        // tau, it's sum_b p(b at tau) * ({1,2} in b) * p((b - 1) -> (2, k))
        tmpmat = eM.topLeftCorner(n, n).transpose() * D_subtend_above.asDiagonal() / num_samples;
        csfs_above.block(2, 0, 1, n) += (tmpmat * sfs_tau).transpose();
    }
    // csfs = csfs_below + csfs_above;
    csfs = csfs_above;
    if (false)
    {
        std::cout << "csfs_below" << std::endl << csfs_below.template cast<double>() << std::endl << std::endl;
        std::cout << "csfs_above" << std::endl << csfs_above.template cast<double>() << std::endl << std::endl;
        std::cout << "csfs" << std::endl << csfs.template cast<double>() << std::endl << std::endl;
    }
    PROGRESS("compute above done");
}

template <typename T>
void ConditionedSFS<T>::fill_matrices(void)
{
    D_subtend_above = Eigen::VectorXd::LinSpaced(n, 1, n).template cast<T>();
    D_subtend_above /= n + 1;
    D_subtend_below = (Eigen::ArrayXd::Ones(n + 1) / Eigen::ArrayXd::LinSpaced(n + 1, 2, n + 2)).template cast<T>();
    D_subtend_below *= 2;
}

template <typename T> 
std::map<int, std::array<Matrix<T>, 3> > ConditionedSFS<T>::matrix_cache;
template <typename T>
std::array<Matrix<T>, 3>& ConditionedSFS<T>::cached_matrices(int n)
{
    if (matrix_cache.count(n) == 0)
    {
        Matrix<double> _Wnbj(n, n), _P_dist(n + 1, n + 1), _P_undist(n + 1, n);
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

        matrix_cache[n] = {_Wnbj.template cast<T>(), _P_dist.template cast<T>(), _P_undist.template cast<T>()};
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
    int num_derivatives = csfs(0,1).derivatives().rows();
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
        int n, int num_samples, const MatrixInterpolator &moran_interp,
        double tau1, double tau2, int numthreads, double theta, 
        double* outsfs)
{
    PiecewiseExponentialRateFunction<double> eta(params);
    // eta.print_debug();
    CSFSManager<double> man(n, moran_interp, numthreads, theta);
    Matrix<double> out = man.compute(eta, num_samples, {tau1, tau2})[0];
    store_sfs_results(out, outsfs);
}

void cython_calculate_sfs_jac(const std::vector<std::vector<double>> params,
        int n, int num_samples, const MatrixInterpolator &moran_interp,
        double tau1, double tau2, int numthreads, double theta, 
        double* outsfs, double* outjac)
{
    PiecewiseExponentialRateFunction<adouble> eta(params);
    // eta.print_debug();
    CSFSManager<adouble> man(n, moran_interp, numthreads, theta);
    Matrix<adouble> out = man.compute(eta, num_samples, {tau1, tau2})[0];
    store_sfs_results(out, outsfs, outjac);
}

template class ConditionedSFS<double>;
template class ConditionedSFS<adouble>;
