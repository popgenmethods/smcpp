#include <gsl/gsl_randist.h>
#include <random>

#include "jcsfs.h"

// Utility functions

inline double scipy_stats_hypergeom_pmf(const int k, const int M, const int n, const int N)
{
    // scipy.stats.hypergeom.pmf(k, M, n, N) = choose(n, k) * choose(M - n, N - k) 
    // gsl_ran_hypergeometric_pdf(unsigned int k, unsigned int n1, unsigned int n2, unsigned int t) = C(n_1, k) C(n_2, t - k) / C(n_1 + n_2, t)
    const int gsl_n1 = n;
    const int gsl_k = k;
    const int gsl_n2 = M - n;
    const int gsl_t = N;
    return gsl_ran_hypergeometric_pdf(gsl_k, gsl_n1, gsl_n2, gsl_t);
}

template <typename T>
Vector<T> undistinguishedSFS(const Matrix<T> &csfs)
{
    int n = csfs.cols() - 1;
    // Total sample size of csfs is n + 2
    Vector<T> ret(n + 2 - 1);
    ret.setZero();
    for (int a = 0; a < 2; ++a)
        for (int b = 0; b < n + 1; ++b)
            if (1 <= a + b and a + b < n + 2)
                ret(a + b - 1) += csfs(a, b);
    return ret;
}

ParameterVector shiftParams(const ParameterVector &model1, const double shift)
{
    const std::vector<adouble> &a = model1[0];
    const std::vector<adouble> &s = model1[1];
    std::vector<adouble> cs(s.size() + 1);
    cs[0] = 0.;
    std::partial_sum(s.begin(), s.end(), cs.begin() + 1);
    cs.back() = INFINITY;
    adouble tshift = shift;
    int ip = std::distance(cs.begin(), std::upper_bound(cs.begin(), cs.end(), tshift)) - 1;
    std::vector<adouble> sp(s.begin() + ip, s.end());
    sp[0] = cs[ip + 1] - shift;
    sp.back() = 1.0;
    std::vector<adouble> ap(a.begin() + ip, a.end());
    return {ap, sp};
}

ParameterVector truncateParams(const ParameterVector params, const double truncationTime)
{
    const std::vector<adouble> &a = params[0];
    const std::vector<adouble> &s = params[1];
    std::vector<adouble> cs(s.size() + 1);
    cs[0] = 0.;
    std::partial_sum(s.begin(), s.end(), cs.begin() + 1);
    cs.back() = INFINITY;
    adouble tt = truncationTime;
    int ip = std::distance(cs.begin(), std::upper_bound(cs.begin(), cs.end(), tt)) - 1;
    std::vector<adouble> sp(s.begin(), s.begin() + ip + 2);
    sp[ip + 1] = truncationTime - cs[ip];
    std::vector<adouble> ap(a.begin(), a.begin() + ip + 2);
    ap.back() = 1e-8; // crash the population to get truncated times.
    return {ap, sp};
}

// Private class methods

template <typename T>
std::map<int, OnePopConditionedSFS<T> > JointCSFS<T>::make_csfs()
{
    std::map<int, OnePopConditionedSFS<T> > ret;
    for (const int &n : {n1, n1 + n2, n1 + n2 - 1, n2 - 2})
        ret.emplace(n, n);
    return ret;
}

template <typename T>
Vector<double> JointCSFS<T>::make_S2()
{
    Vector<double> v(n1 + 2);
    for (int i = 0; i < n1 + 2; ++i)
        v(i) = (double)i / (double)(n1 + 1);
    return v;
}

template <typename T>
void JointCSFS<T>::jcsfs_helper_tau_below_split(const int m, 
        const double t1, const double t2, const T weight)
{
    assert(t1 < t2 <= split);
    assert(a1 == 2);
    DEBUG << "jcsfs_below t1:" << t1 << " t2:" << t2;
    const PiecewiseConstantRateFunction<T> eta(params1, {});

    const ParameterVector params1_trunc = truncateParams(params1, split);
    const PiecewiseConstantRateFunction<T> eta1_trunc(params1_trunc, {t1, t2});
    const Matrix<T> trunc_csfs = csfs.at(n1).compute(eta1_trunc)[0];
    for (int i = 0; i < a1 + 1; ++i)
        for (int j = 0; j < n1 + 1; ++j)
        {
            assert(trunc_csfs(i, j) > -1e-8); // truncation may lead to small negative values.
            if (trunc_csfs(i, j) > 0)
                tensorRef(m, i, j, 0) = weight * trunc_csfs(i, j);
        }

    const ParameterVector params1_shift = shiftParams(params1, split);
    const PiecewiseConstantRateFunction<T> eta1_shift(params1_shift, {0., INFINITY});
    Matrix<T> sfs_above_split = undistinguishedSFS(csfs.at(n1 + n2 - 1).compute(eta1_shift)[0]);
    Matrix<T> eMn10_avg(n1 + 2, n1 + 1), eMn12_avg(n1 + 2, n1 + 1);
    eMn10_avg.setZero();
    eMn12_avg.setZero();
    std::mt19937 gen;
    for (int k = 0; k < K; ++k)
    {
        // FIXME do something with seeding.
        T t = eta.random_time(1., t1, t2, gen);
        T Rt = eta.R(t);
        Matrix<T> A = Mn1.expM(Rts1 - Rt);
        Matrix<T> B = Mn10.expM(Rt);
        Matrix<T> C = Mn12.expM(Rt);
        eMn10_avg += (A * S0.template cast<T>().asDiagonal()).leftCols(n1 + 1) * B;
        eMn12_avg += (A * S2.template cast<T>().asDiagonal()).rightCols(n1 + 1) * C;
        Matrix<double> eMn10_d = eMn10_avg.template cast<double>();
        Matrix<double> eMn12_d = eMn12_avg.template cast<double>();
        if (eMn10_d.minCoeff() < -1e-8)
            throw std::runtime_error("emn10 is wrong");
        if (eMn12_d.minCoeff() < -1e-8)
            throw std::runtime_error("emn12 is wrong");
    }
    eMn10_avg /= (double)K;
    eMn12_avg /= (double)K;
    // Now moran down
    for (int b1 = 0; b1 < n1 + 1; ++b1)
        for (int b2 = 0; b2 < n2 + 1; ++b2)
            for (int nseg = 1; nseg < n1 + n2 + 1; ++nseg)
                for (int np1 = std::max(nseg - n2, 0); np1 < std::min(nseg, n1) + 1; ++np1)
                {
                    int np2 = nseg - np1;
                    double h = scipy_stats_hypergeom_pmf(np1, n1 + n2, nseg, n1);
                    tensorRef(m, 0, b1, b2) += weight * h * sfs_above_split(nseg - 1) * eMn10_avg(np1, b1) * eMn2(np2, b2);
                    tensorRef(m, 2, b1, b2) += weight * h * sfs_above_split(nseg - 1) * eMn12_avg(np1, b1) * eMn2(np2, b2);
                    check_negative(tensorRef(m, 0, b1, b2));
                    check_negative(tensorRef(m, 2, b1, b2));
                }
}

template <typename T>
void JointCSFS<T>::jcsfs_helper_tau_above_split(const int m, 
        const double t1, const double t2, const T weight)
{
    assert(split <= t1 < t2);
    assert(a1 == 2);
    DEBUG << "jcsfs_above t1:" << t1 << " t2:" << t2;
    // Shift eta1 back by split units in time 
    PiecewiseConstantRateFunction<T> shifted_eta1(shiftParams(params1, split), {t1 - split, t2 - split});
    Matrix<T> rsfs = csfs.at(n1 + n2).compute(shifted_eta1)[0];

    for (int b1 = 0; b1 < n1 + 1; ++b1)
        for (int b2 = 0; b2 < n2 + 1; ++b2)
            for (int nseg = 0; nseg < n1 + n2 + 1; ++nseg)
                for (int np1 = std::max(nseg - n2, 0); nseg < std::min(nseg, n1) + 1; ++nseg)
                {
                    int np2 = nseg - np1;
                    // scipy.stats.hypergeom.pmf(np1, n1 + n2, nseg, n1)  = choose(nseg, np1) * choose(n1 + n2 - nseg, n1 - np1)
                    // gsl_ran_hypergeometric_pdf(unsigned int k, unsigned int n1, unsigned int n2, unsigned int t) 
                    double h = scipy_stats_hypergeom_pmf(np1, n1 + n2, nseg, n1);
                    for (int i = 0; i < 3; ++i)
                    {
                        int ind = i * (n1 + 1) * (n2 + 1) + b1 * (n2 + 1) + b2;
                        tensorRef(m, i, b1, b2) += weight * h * rsfs(i, nseg) * eMn1[i](np1, b1) * eMn2(np2, b2);
                        check_negative(tensorRef(m, i, b1, b2));
                    }
                }
     
    // pop 1, below split
    Matrix<T> sfs_below = csfs.at(n1).compute(*eta1)[0];
    for (int i = 0; i < a1 + 1; ++i)
        for (int j = 0; j < n1 + 1; ++j)
        {
            assert(sfs_below(i, j) > -1e-8);
            if (sfs_below(i, j) > 0)
                tensorRef(m, i, j, 0) += weight * sfs_below(i, j);
            check_negative(tensorRef(m, i, j, 0));
        }

    // pop2, below split
    if (n2 == 1)
        tensorRef(m, 0, 0, 1) += weight * split;
    if (n2 > 1)
    {
        ParameterVector params2_trunc = truncateParams(params2, split);
        const PiecewiseConstantRateFunction<T> eta2_trunc(params2_trunc, {0., INFINITY});
        Vector<T> rsfs_below_2 = undistinguishedSFS(csfs.at(n2 - 2).compute(eta2_trunc)[0]);
        assert(rsfs_below_2.size() == n2 - 1);
        for (int i = 0; i < n2 - 1; ++i)
        {
            tensorRef(m, 0, 0, i + 1) += weight * rsfs_below_2(i);
            check_negative(tensorRef(m, 0, 0, i + 1));
        }
    }
}

template <typename T>
std::vector<Matrix<T> > JointCSFS<T>::compute(const PiecewiseConstantRateFunction<T>&) const
{
    return J;
}


// Public class methods
template <typename T>
void JointCSFS<T>::pre_compute(
        const ParameterVector &params1, 
        const ParameterVector &params2, 
        double split)
{
    this->split = split;
    this->params1 = params1;
    this->params2 = params2;
    eta1.reset(new PiecewiseConstantRateFunction<T>(params1, {split - 1e-6, split + 1e-6}));
    eta2.reset(new PiecewiseConstantRateFunction<T>(params2, {}));
    Rts1 = eta1->R(split);
    Rts2 = eta2->R(split);
    eMn1[0] = Mn10.expM(Rts1);
    eMn1[1] = Mn11.expM(Rts1);
    eMn1[2] = eMn1[0].reverse();
    eMn2 = Mn2.expM(Rts2);
#pragma omp parallel for
    for (int m = 0; m < M; ++m)
    {
        J[m].setZero();
        double t1 = hidden_states[m], t2 = hidden_states[m + 1];
        if (t1 < t2 and t2 <= split)
            jcsfs_helper_tau_below_split(m, t1, t2, 1.); 
        else if (split <= t1 and t1 < t2)
            jcsfs_helper_tau_above_split(m, t1, t2, 1.); 
        else
        {
            T eR1t1 = exp(-eta1->R(t1)), 
              eR1t2 = exp(-eta1->R(t2));
            T w = (exp(-Rts1) - eR1t2) / (eR1t1 - eR1t2);
            jcsfs_helper_tau_below_split(m, t1, split, 1. - w);
            jcsfs_helper_tau_above_split(m, split, t2, w);
        }
    }
}

// Instantiate necessary templates
template class JointCSFS<double>;
template class JointCSFS<adouble>;

