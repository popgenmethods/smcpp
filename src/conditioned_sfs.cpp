#include <cfenv>
#include "conditioned_sfs.h"

#define EIGEN_NO_AUTOMATIC_RESIZING 1

std::random_device rd;  // only used once to initialise (seed) engine

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

inline adouble fmin(adouble a, adouble b)
{
#ifdef AUTODIFF
    return (a + b - Eigen::abs(a - b)) / 2;
#else
    return (a + b - std::abs(a - b)) / 2;
#endif
}

ConditionedSFS::ConditionedSFS(PiecewiseExponential eta, int n) :
    gen(rd()),
    eta(eta), n(n),
    D_subtend_above(n, n), D_not_subtend_above(n, n),
	D_subtend_below(n + 1, n + 1), D_not_subtend_below(n + 1, n + 1),
	Wnbj(n, n), P_dist(n + 1, n + 1), 
    P_undist(n + 1, n), tK(n + 1, n + 1), csfs(3, n + 1), csfs_above(3, n + 1), csfs_below(3, n + 1), ETnk_below(n + 1, n + 1)
{
    // feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT & ~FE_UNDERFLOW);
    // printf("seed: %u\n", seed);
    fill_matrices();
}

double ConditionedSFS::unif(void)
{
    return std::uniform_real_distribution<double>{0.0, 1.0}(gen);
}

double ConditionedSFS::exp1(void)
{
    return std::exponential_distribution<double>{1.0}(gen);
}

double ConditionedSFS::exp1_conditional(double a, double b)
{
    // If X ~ Exp(1),
    // P(X < x | a <= X <= b) = (e^-a - e^-x) / (e^-a - e^-b)
    // so P^-1(y) = -log(e^-a - (e^-a - e^-b) * y)
    //            = -log(e^-a(1 - (1 - e^-(b-a)) * y)
    //            = a - log(1 - (1 - e^-(b-a)) * y)
    //            = a - log(1 + expm1(-(b-a)) * y)
    if (isinf(b))
        return a - log1p(-unif());
    else
        return a - log1p(expm1(-(b - a)) * unif());
}

void ConditionedSFS::compute_ETnk_below(const AdVector &etjj)
{
    ETnk_below.setZero();
    ETnk_below.diagonal() = etjj;
    for (int nn = 2; nn < n + 3; ++nn)
    {
        for (int k = nn - 1; k > 1; --k)
        {
            ETnk_below(nn - 2, k - 2) = ETnk_below(nn - 3, k - 2) -
                (double)(k + 2) * (k - 1) / (nn + 1) / (nn - 2) * ETnk_below(nn - 2, k - 1);
            ETnk_below(nn - 2, k - 2) /= 1.0 - (double)(k + 1) * (k - 2) / (nn + 1) / (nn - 2);
        }
    }
}

std::thread ConditionedSFS::compute_threaded(int S, int M, const std::vector<double> &ts, 
        const std::vector<Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>> &expM,
        double t1, double t2)
{
    return std::thread(&ConditionedSFS::compute, this, S, M, ts, expM, t1, t2);    
}

// Calculate sfs and derivatives for rate function 
//     eta(t) = a[k] * exp(b[k] * (t - t[k])), t[k] <= t < t[k + 1]
// where t[k] = s[1]**2 + ... + s[k]**2.
// conditioned on two lineages coalescing between tau1 and tau2
void ConditionedSFS::compute(int S, int M, const std::vector<double> &ts, 
        const std::vector<Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>> &expM,
        double t1, double t2)
{
    feenableexcept(FE_INVALID | FE_OVERFLOW);
    adouble tau;
    // There are n + 2 (undistinguished + distinguished) at time 0.
    // Above tau there are between 2 and n + 1.
	// For some reason it does not like initializing these as AdVectors. So use
	// the matrix class instead.
    AdMatrix tjj_above(n, 1), tjj_below(n + 1, 1), sfs_tau(n, 1), etnk(n + 1, 1);
    AdMatrix etnk_avg(n + 1, 1), tjj_below_avg(n + 1, 1), tjj_above_avg(n, 1);
	Eigen::MatrixXd eM(n + 1, n + 1), trans1, trans2;
    std::map<int[2], double> etnk_memo;
    int m, rate_above, rate_below, ei;
    double coef, y;
    etnk_avg.setZero();
    tjj_below_avg.setZero();
    Eigen::VectorXd der;
	Eigen::MatrixXd tmpmat;
	AdMatrix tmpvec;

	// Mixing constants with adoubles causes problems because the library
	// doesn't know how to allocate the VectorXd of derivatives(). 
	// Here, we do it by hand.
	adouble zero = 0;
    auto d = Eigen::VectorXd(eta.num_derivatives());
    d.fill(0);
	zero.derivatives() = d;
    csfs.fill(zero);
    csfs_above.fill(zero);
    csfs_below.fill(zero);

    // Fill the vector of interpolating rate matrices
    // Simulate ETjj above and below tau
    for (m = 0; m < M; ++m)
    {
        _DEBUG(std::cout << m << " " << std::flush);
        y = exp1_conditional(t1, t2);
        ei = insertion_point(y, ts, 0, ts.size());
        tau = eta.inverse_rate(y, zero, 1);
        // _DEBUG(std::cout << tau.derivatives() << " " << std::endl);
        der = tau.derivatives();
        tjj_below.fill(zero);
        tjj_above.fill(zero);
        tjj_below(0, 0) = tau;
        // etjj above and below
        for (int j = 2; j < n + 2; ++j)
        {
            rate_above = (double)(j * (j - 1) / 2);
            rate_below = (double)((j + 1) * j / 2 - 1);
            for (int s = 0; s < S; ++s)
            {
                tjj_below(j - 1, 0) += fmin(tau, eta.inverse_rate(exp1(), zero, rate_below)) / S;
                tjj_above(j - 2, 0) += eta.inverse_rate(exp1(), tau, rate_above) / S;
            }
        }
        // Compute sfs below using ETnk recursion
        compute_ETnk_below(tjj_below);
		// Unfortunately, mixed-products have to be done using lazyProduct() for now
        etnk = tK.lazyProduct(ETnk_below.row(n).transpose());
        etnk_avg += etnk / M;
        tjj_below_avg += tjj_below / M;
        tjj_above_avg += tjj_above / M;
        csfs_below.block(0, 1, 1, n) += etnk.transpose().lazyProduct(D_not_subtend_below * P_undist / M);
        csfs_below.block(1, 0, 1, n + 1) += etnk.transpose().lazyProduct(D_subtend_below * P_dist / M);
        // Compute sfs above using polanski-kimmel + matrix exponential
        // Get the correct linear-interpolated matrix exponential
        coef = (y - ts[ei]) / (ts[ei + 1] - ts[ei]);
        eM = (1 - coef) * expM[ei] + coef * expM[ei + 1];
        // Wnbj is the Polanski-Kimmel matrix of coefficients W_(n + 1, b, j)
        sfs_tau = Wnbj.lazyProduct(tjj_above); // n-vector; sfs_tau[b] = prob. lineage subtends b + 1
        // If a = 0 (neither distinguished lineage is derived) then immediately after
        // tau, it's sum_b p(|B|=b at tau) * ({1,2} not a subset of B) * p(|B| -> (0, k))
        // Recall that the rate matrix for the a = 0 case is the reverse of the a = 2 case,
        // which is what gets passed in.
        tmpmat = eM.reverse().bottomRightCorner(n, n).transpose() * D_not_subtend_above / M;
		csfs_above.block(0, 1, 1, n) += tmpmat.lazyProduct(sfs_tau).transpose();
        // If a = 2 (both distinguished lineages derived) then immediately after
        // tau, it's sum_b p(b at tau) * ({1,2} in b) * p((b - 1) -> (2, k))
        tmpmat = eM.topLeftCorner(n, n).transpose() * D_subtend_above / M;
        csfs_above.block(2, 0, 1, n) += tmpmat.lazyProduct(sfs_tau).transpose();
    }
    csfs = csfs_below + csfs_above;
}

void store_sfs_results(const AdMatrix &csfs, double* outsfs, double* outjac)
{
    int n = csfs.cols();
    int num_derivatives = csfs(0,1).derivatives().rows();
    Eigen::VectorXd d;
    int m = 0;
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> _outsfs(outsfs, 3, n);
    _outsfs = csfs.cast<double>();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < n; ++j)
        {
            d = csfs(i, j).derivatives();
            assert(d.rows() == num_derivatives);
            /*
            for (int k = 0; k < eta->K(); ++k)
                printf("i:%i j:%i dsfs/da[%i]:%f dsfs/db[%i]:%f dsfs/ds[%i]:%f\n", i, j, k, d(k), k, d(eta->K() + k), k, d(k + eta->K() * 2));
                */
            for (int k = 0; k < num_derivatives; ++k)
                outjac[m++] = d(k);
        }
}

void ConditionedSFS::set_seed(long long seed)
{
    gen.seed(seed);
}

void ConditionedSFS::fill_matrices(void)
{
	Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
	Eigen::MatrixXd I1 = Eigen::MatrixXd::Identity(n + 1, n + 1);
    // Construct some matrices that will be used later on
    D_subtend_above.setZero();
    D_subtend_above.diagonal() = Eigen::VectorXd::LinSpaced(n, 1, n) / (n + 1);
	D_not_subtend_above = I - D_subtend_above;

    D_subtend_below.setZero();
    for (int k = 2; k < n + 3; ++k)
        D_subtend_below.diagonal()(k - 2) = 2. / k;
	D_not_subtend_below = I1 - D_subtend_below;

    tK.setZero();
    tK.diagonal() = Eigen::VectorXd::LinSpaced(n + 1, 2, n + 2);

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

AdMatrix ConditionedSFS::matrix(void) const
{
    return csfs;
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

AdMatrix average_csfs(std::vector<ConditionedSFS> &csfs, double theta)
{
    AdMatrix ret = AdMatrix::Zero(csfs[0].matrix().rows(), csfs[0].matrix().cols());
    int m = 0;
    for (const ConditionedSFS &c : csfs)
    {
        ret += c.matrix();
        // std::cout << std::endl << c.matrix().cast<double>() << std::endl << std::endl;
        ++m;
    }
    ret /= (double)m;
    // std::cout << ret.cast<double>() << std::endl << std::endl;
    ret *= theta;
    ret(0, 0) = 1.0 - ret.sum();
    return ret;
}

AdMatrix calculate_sfs(PiecewiseExponential eta, int n, int S, int M, const std::vector<double> &ts, 
        const std::vector<double*> &expM, double t1, double t2, int numthreads, double theta)
{
    std::vector<ConditionedSFS> csfs;
    std::vector<std::thread> t;
    int P = expM.size();
    std::vector<Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>> _expM;
    for (int p = 0; p < P; ++p)
        _expM.emplace_back(expM[p], n + 1, n + 1);
    for (int i = 0; i < numthreads; ++i)
        csfs.emplace_back(eta, n);
    for (auto &c : csfs)
        t.push_back(c.compute_threaded(S, M, ts, _expM, t1, t2));
    std::for_each(t.begin(), t.end(), [](std::thread &t) {t.join();});
    AdMatrix ret = average_csfs(csfs, theta);
    // std::cout << std::endl << ret.cast<double>() << std::endl << std::endl;
    return ret;
}
/*
int main(int argc, char** argv)
{
    int K = 2;
    int n = 50;
    double sqrt_a[2] = {1.0, 0.5};
    double b[2] = {.01, -.001};
    double sqrt_s[2] = {0.0, 1.0};
    double outsfs[3 * (n + 1)];
    double outjac[3 * 2 * 3 * (n + 1)];
    double expm[(n + 1) * (n + 1)];
    int m = 0;
    for (int i = 0; i <= n; ++i)
        for (int j = 0; j <= n; ++j)
            expm[m++] = (int)(i == j);
    std::vector<double*> E = {expm, expm, expm, expm};
    std::vector<int> ei(1000, 1);
    std::vector<double> t(1000);
    double x = 0.01;
    for (int m = 0; m < 1000; ++m)
        t[m] = (x += 0.01);
    std::vector<double> y(1000, 0.2);

    ConditionedSFS csfs(K, n, sqrt_a, b, sqrt_s, outsfs, outjac);
    csfs.compute(100, 100, &y[0], E, &ei[0], &t[0], 1e-6);
}

*/
