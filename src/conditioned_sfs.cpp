#include "conditioned_sfs.h"

std::mt19937 sfs_gen;

std::map<int, below_coeff> ConditionedSFSBase::below_coeffs_memo;

below_coeff compute_below_coeffs(int n)
{
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
            mnew[k - 2] -= mnew[k - 1] * mpq_class((k + 1) * (k - 2), denom);
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
                    return std::max(pp, (double)log10(abs(r)));
                });
    ret.prec = mpfr::digits2bits(p) + 20;
    std::vector<std::valarray<mpfr::mpreal>> v(n + 1, std::valarray<mpfr::mpreal>(n + 1));
    for (int k = 2; k < n + 3; ++k)
        std::transform(std::begin(mlast[k - 2]), std::end(mlast[k - 2]), std::begin(v[k - 2]),
                [&ret] (mpq_class &x)
                {
                    return mpfr::mpreal(x.get_mpq_t(), ret.prec);
                });
    ret.coeffs = v;
    return ret;
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

double pnkb_dist(int n, int m, int l1)
{
    // Probability that lineage 1 has size |L_1|=l1 below tau,
    // the time at which 1 and 2 coalesce, when there are k 
    // undistinguished lineages remaining, in a sample of n
    // undistinguished (+2 distinguished) lineages overall.
    double ret = l1 * (n + 2 - l1) / (double) binom(n + 3, m + 3);
    if (m > 0)
        ret *= (n + 1 - l1) * (double)binom(n - l1, m - 1) / m / (m + 1);
    return ret;
}

double pnkb_undist(int n, int m, int l3)
{
    // Probability that undistinguished lineage has size |L_1|=l1 below tau,
    // the time at which 1 and 2 coalesce, when there are k 
    // undistinguished lineages remaining, in a sample of n
    // undistinguished (+2 distinguished) lineages overall.
    assert(m > 0);
    double ret = (n + 3 - l3) * (n + 2 - l3) * (n + 1 - l3) / (double) binom(n + 3, m + 3);
    if (m == 1)
        ret /= 6.0;
    else
        ret *= (n - l3) * (double)binom(n - l3 - 1, m - 2) / (m - 1) / m / (m + 1) / (m + 2);
    return ret;
}


template <typename T>
ConditionedSFS<T>::ConditionedSFS(const PiecewiseExponentialRateFunction<T> eta, int n, MatrixInterpolator moran_interp) :
    eta(eta), n(n), moran_interp(moran_interp), 
    D_subtend_above(n, n), D_not_subtend_above(n, n),
	D_subtend_below(n + 1, n + 1), D_not_subtend_below(n + 1, n + 1),
	Wnbj(n, n), P_dist(n + 1, n + 1), 
    P_undist(n + 1, n), tK(n + 1, n + 1), csfs(3, n + 1), 
    csfs_above(3, n + 1), csfs_below(3, n + 1), bc(compute_below_coeffs(n))
{
    long long seed = std::uniform_int_distribution<long long>{}(sfs_gen);
    gen.seed(seed);
    fill_matrices();
    if (below_coeffs_memo.count(n) == 0)
        below_coeffs_memo[n] = compute_below_coeffs(n);
    bc = below_coeffs_memo[n];
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
    std::cout << bc.prec << " bits of precision needed..." << std::endl;
    mpfr::mpreal::set_default_prec(bc.prec);
    Vector<T> ret = Vector<T>::Zero(n + 1);
    std::valarray<mpreal_wrapper<T>> v(n + 1);
    for (int i = 0; i < n + 1; ++i)
        v[i] = etjj[i];
    for (int i = 0; i < n + 1; ++i)
        for (int j = 0; j < n + 1; ++j)
            ret(i) += (bc.coeffs[i][j] * v[j]).toDouble();
    return tK * ret;
}

template <typename T>
std::thread ConditionedSFS<T>::compute_threaded(int num_samples, T t1, T t2)
{
    return std::thread(&ConditionedSFS::compute, this, num_samples, t1, t2);    
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

int get_nd(double x) { return 0; }
int get_nd(adouble x) { return x.derivatives().size(); }

template <typename T>
void ConditionedSFS<T>::compute(int num_samples, T t1, T t2)
{
    // feenableexcept(FE_INVALID | FE_OVERFLOW);
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
    eta.print_debug();

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
        tjj_below(0) = tau;
        mpfr_tjj_below[0] = mpreal_wrapper<T>(tau);
        for (int j = 2; j < n + 2; ++j)
        {
            double rate = (double)((j + 1) * j / 2 - 1);
            mpfr_tjj_below[j - 1] = eta.mpfr_tjj_integral(rate, zero, tau, zero);
            rate = (double)(j * (j - 1) / 2);
            tjj_above(j - 2) = eta.tjj_integral(rate, tau, INFINITY, y);
        }
        etnk_below = compute_etnk_below(mpfr_tjj_below);
        csfs_below.block(0, 1, 1, n) += etnk_below.transpose() * D_not_subtend_below * P_undist / num_samples;
        csfs_below.block(1, 0, 1, n + 1) += etnk_below.transpose() * D_subtend_below * P_dist / num_samples;
        // Compute sfs above using polanski-kimmel + matrix exponential
        // Get the correct linear-interpolated matrix exponential
        eM = moran_interp.interpolate<T>(y);
        // Wnbj is the Polanski-Kimmel matrix of coefficients W_(n + 1, b, j)
        sfs_tau = Wnbj * tjj_above; // n-vector; sfs_tau[b] = prob. lineage subtends b + 1
        // If a = 0 (neither distinguished lineage is derived) then immediately after
        // tau, it's sum_b p(|B|=b at tau) * ({1,2} not a subset of B) * p(|B| -> (0, k))
        // Recall that the rate matrix for the a = 0 case is the reverse of the a = 2 case,
        // which is what gets passed in.
        tmpmat = eM.reverse().bottomRightCorner(n, n).transpose() * D_not_subtend_above / num_samples;
		csfs_above.block(0, 1, 1, n) += (tmpmat * sfs_tau).transpose();
        // If a = 2 (both distinguished lineages derived) then immediately after
        // tau, it's sum_b p(b at tau) * ({1,2} in b) * p((b - 1) -> (2, k))
        tmpmat = eM.topLeftCorner(n, n).transpose() * D_subtend_above / num_samples;
        csfs_above.block(2, 0, 1, n) += (tmpmat * sfs_tau).transpose();
    }
    csfs = csfs_below + csfs_above;
    if (false)
    {
        std::cout << "csfs_below" << std::endl << csfs_below.template cast<double>() << std::endl << std::endl;
        std::cout << "csfs_above" << std::endl << csfs_above.template cast<double>() << std::endl << std::endl;
        std::cout << "csfs" << std::endl << csfs.template cast<double>() << std::endl << std::endl;
    }
}

template <typename T>
void ConditionedSFS<T>::fill_matrices(void)
{
	Matrix<T> I = Matrix<T>::Identity(n, n);
	Matrix<T> I1 = Matrix<T>::Identity(n + 1, n + 1);
    // Construct some matrices that will be used later on
    D_subtend_above.setZero();
    D_subtend_above.diagonal() = Eigen::VectorXd::LinSpaced(n, 1, n).template cast<T>() / (n + 1);
	D_not_subtend_above = I - D_subtend_above;

    D_subtend_below.setZero();
    for (int k = 2; k < n + 3; ++k)
        D_subtend_below.diagonal()(k - 2) = 2. / k;
	D_not_subtend_below = I1 - D_subtend_below;

    tK.setZero();
    tK.diagonal() = Eigen::VectorXd::LinSpaced(n + 1, 2, n + 2).template cast<T>();

    // Calculate the Polanski-Kimmel matrix
    // TODO: this could be sped up by storing the matrix outside of the class
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

template <typename T>
Matrix<T> ConditionedSFS<T>::average_csfs(std::vector<ConditionedSFS<T>> &csfs, double theta)
{
    Matrix<T> ret = Matrix<T>::Zero(csfs[0].matrix().rows(), csfs[0].matrix().cols());
    int m = 0;
    for (const ConditionedSFS<T> &c : csfs)
    {
        ret += c.matrix();
        ++m;
    }
    ret /= (double)m;
    T tauh = ret.sum();
    ret *= -expm1(-theta * tauh) / tauh;
    ret(0, 0) = exp(-theta * tauh);
    // ret *= theta;
    // ret(0, 0) = 1. - ret.sum();
    return ret;
}

template <typename T>
Matrix<T> ConditionedSFS<T>::calculate_sfs(const PiecewiseExponentialRateFunction<T> &eta, int n, int num_samples, 
        const MatrixInterpolator &moran_interp, double tau1, double tau2, int numthreads, double theta)
{
    // eta.print_debug();
    auto R = eta.getR();
    std::vector<ConditionedSFS<T>> csfs;
    std::vector<std::thread> t;
    T t1 = (*R)(tau1);
    T t2;
    if (std::isinf(tau2))
        t2 = INFINITY;
    else
        t2 = (*R)(tau2);
    std::vector<Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>> _expM;
    if (numthreads == 0)
    {
        csfs.emplace_back(eta, n, moran_interp);
        csfs.back().compute(num_samples, t1, t2);
    }
    else
    {
        for (int i = 0; i < numthreads; ++i)
            csfs.emplace_back(eta, n, moran_interp);
        for (auto &c : csfs)
            t.push_back(c.compute_threaded(num_samples, t1, t2));
        std::for_each(t.begin(), t.end(), [](std::thread &t) {t.join();});
    }
    Eigen::Matrix<T, 3, Eigen::Dynamic> ret = ConditionedSFS<T>::average_csfs(csfs, theta);
    if (ret(0,0) <= 0.0 or ret(0.0) >= 1.0)
    {
        std::cout << ret.template cast<double>() << std::endl << std::endl;
        std::cout << t1 << " " << t2 << std::endl << std::endl;
        std::cerr << "sfs is no longer a probability distribution. branch lengths are too long." << std::endl;
    }
    return ret;
}

void set_seed(long long seed)
{
    sfs_gen.seed(seed);
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
    RATE_FUNCTION<double> eta(params);
    // eta.print_debug();
    Matrix<double> out = ConditionedSFS<double>::calculate_sfs(eta, n, num_samples, moran_interp, tau1, tau2, numthreads, theta);
    store_sfs_results(out, outsfs);
}

void cython_calculate_sfs_jac(const std::vector<std::vector<double>> params,
        int n, int num_samples, const MatrixInterpolator &moran_interp,
        double tau1, double tau2, int numthreads, double theta, 
        double* outsfs, double* outjac)
{
    RATE_FUNCTION<adouble> eta(params);
    // eta.print_debug();
    Matrix<adouble> out = ConditionedSFS<adouble>::calculate_sfs(eta, n, num_samples, moran_interp, tau1, tau2, numthreads, theta);
    store_sfs_results(out, outsfs, outjac);
}

template class ConditionedSFS<double>;
template class ConditionedSFS<adouble>;
