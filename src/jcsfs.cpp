#include <gsl/gsl_randist.h>
#include <random>

#include "jcsfs.h"

// Utility functions
double hyp(const int k, const int M, const int n, const int N)
{
   // scipy.stats.hypergeom.pmf(k, M, n, N) = choose(n, k) * choose(M - n, N - k)
   // gsl_ran_hypergeometric_pdf(unsigned int k, unsigned int n1, unsigned int n2, unsigned int t) = C(n_1, k) C(n_2, t - k) / C(
   const int gsl_n1 = n;
   const int gsl_k = k;
   const int gsl_n2 = M - n;
   const int gsl_t = N;
   return gsl_ran_hypergeometric_pdf(gsl_k, gsl_n1, gsl_n2, gsl_t);
}

template <typename T>
Matrix<double> JointCSFS<T>::make_hyp1()
{
    Matrix<double> ret(n1 + 1, n1 + n2 + 1);
    ret.setZero();
    for (int b1 = 0; b1 <= n1; ++b1)
        for (int b2 = 0; b2 <= n2; ++b2)
            for (int nseg = 0; nseg <= n1 + n2; ++nseg)
                for (int np1 = std::max(nseg - n2, 0); np1 <= std::min(nseg, n1); ++np1)
                    ret(np1, nseg) = hyp(np1, n1 + n2, nseg, n1);
    return ret;
}

template <typename T>
Matrix<double> JointCSFS<T>::make_hyp2()
{
    Matrix<double> ret(n1 + 2, n1 + n2);
    ret.setZero();
    for (int b1 = 0; b1 <= n1; ++b1)
        for (int b2 = 0; b2 <= n2; ++b2)
            for (int nseg = 1; nseg <= n1 + n2; ++nseg)
                for (int np1 = std::max(nseg - n2, 0); np1 <= std::min(nseg, n1 + 1); ++np1)
                    ret(np1, nseg - 1) = hyp(np1, n1 + n2 + 1, nseg, n1 + 1);
    return ret;
}

template <typename T>
double JointCSFS<T>::scipy_stats_hypergeom_pmf(const int k, const int M, const int n, const int N)
{
    if (M == n1 + n2)
        return hyp1(k, n);
    else if (M == n1 + n2 + 1)
        return hyp2(k, n - 1);
    else
        throw std::runtime_error("bad hyp");
}

template <typename T>
Vector<T> undistinguishedSFS(const Matrix<T> &csfs)
{
    int n = csfs.cols() - 1;
    // Total sample size of csfs is n + 2
    Vector<T> ret(n + 2 - 1);
    ret.setZero();
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < n + 1; ++b)
            if (1 <= a + b and a + b < n + 2)
                ret(a + b - 1) += csfs(a, b);
    return ret;
}

// Private class methods

template <typename T>
std::map<int, OnePopConditionedSFS<T> > JointCSFS<T>::make_csfs()
{
    std::map<int, OnePopConditionedSFS<T> > ret;
    for (const int &n : {n1, n1 - 1, n2 - 1, n1 + n2, n1 + n2 - 1, n2 - 2})
        if (n >= 0)
            ret.emplace(n, n);
    return ret;
}

template <typename T>
Vector<double> JointCSFS<T>::arange(int a, int b) const
{
    Vector<double> v(b - a);
    std::iota(v.data(), v.data() + (b - a), a);
    return v;
}

template <typename T>
void JointCSFS<T>::jcsfs_helper_tau_below_split(const int m, 
        const double t1, const double t2, const T weight)
{
    assert(t1 < t2 <= split);
    assert(a1 == 2);
    DEBUG1 << "jcsfs_below t1:" << t1 << " t2:" << t2;
    const PiecewiseConstantRateFunction<T> eta(params1, {});

    const ParameterVector params1_trunc = truncateParams(params1, split);
    const PiecewiseConstantRateFunction<T> eta1_trunc(params1_trunc, {t1, t2});
    const Matrix<T> trunc_csfs = csfs.at(n1).compute(eta1_trunc)[0];
    for (int i = 0; i < a1 + 1; ++i)
        for (int j = 0; j < n1 + 1; ++j)
        {
            assert(trunc_csfs(i, j) > -1e-8); // truncation may lead to small negative values.
            if (trunc_csfs(i, j) > 0)
                tensorRef(m, i, j, 0, 0) = weight * trunc_csfs(i, j);
        }
    const Vector<T> trunc_sfs = undistinguishedSFS(trunc_csfs);
    T Et = Sn1.transpose().template cast<T>() * trunc_sfs;
    tensorRef(m, 2, n1, 0, 0) = split - Et;
    tensorRef(m, 2, n1, 0, 0) *= weight;

    // Above split, then moran down
    const ParameterVector params1_shift = shiftParams(params1, split);
    const PiecewiseConstantRateFunction<T> eta1_shift(params1_shift, {0., INFINITY});
    const Vector<T> sfs_above_split = undistinguishedSFS(csfs.at(n1 + n2 - 1).compute(eta1_shift)[0]);
    Matrix<T> eMn10_avg(n1 + 2, n1 + 1), eMn12_avg(n1 + 2, n1 + 1);
    eMn10_avg.fill(eta1_shift.zero());
    eMn12_avg.fill(eta1_shift.zero());
    std::mt19937 gen;
    const T cRts1 = Rts1;
    for (int k = 0; k < K; ++k)
    {
        // FIXME do something with seeding.
        const T t = eta.random_time(1., t1, t2, gen);
        const T Rt = eta.R(t);
        const Matrix<T> A = togetherM.Mn1p1.expM(cRts1 - Rt);
        const Matrix<T> B = togetherM.Mn10.expM(Rt);
        const Matrix<T> C = togetherM.Mn12.expM(Rt);
        const Matrix<T> Mn10 = (A * S0.template cast<T>().asDiagonal()).leftCols(n1 + 1) * B;
        const Matrix<T> Mn12 = (A * S2.template cast<T>().asDiagonal()).rightCols(n1 + 1) * C;
        eMn10_avg += Mn10;
        eMn12_avg += Mn12;
    }
    eMn10_avg /= (double)K;
    eMn12_avg /= (double)K;
    Matrix<T> const& c_eMn10_avg = eMn10_avg;
    Matrix<T> const& c_eMn12_avg = eMn12_avg;
    Matrix<T> const& c_eMn2 = eMn2;
    // Now moran down
#pragma omp parallel for collapse(2)
    for (int b1 = 0; b1 <= n1; ++b1)
    {
        for (int b2 = 0; b2 <= n2; ++b2)
        {
            for (int nseg = 1; nseg <= n1 + n2; ++nseg)
            {
                for (int np1 = std::max(nseg - n2, 0); np1 <= std::min(nseg, n1 + 1); ++np1)
                {
                    const int np2 = nseg - np1;
                    const double h = scipy_stats_hypergeom_pmf(np1, n1 + n2 + 1, nseg, n1 + 1);
                    T x = sfs_above_split(nseg - 1);
                    x *= c_eMn2(np2, b2);
                    x *= h;
                    x *= weight;
                    T x0 = x * c_eMn10_avg(np1, b1);
                    T x2 = x * c_eMn12_avg(np1, b1);
                    tensorRef(m, 0, b1, 0, b2) += x0;
                    tensorRef(m, 2, b1, 0, b2) += x2;
                }
            }
        }
    }
}

template <typename T>
void JointCSFS<T>::jcsfs_helper_tau_above_split(const int m, 
        const double t1, const double t2, const T weight)
{
    assert(split <= t1 < t2);
    assert(a1 == 2);
    DEBUG1 << "jcsfs_above t1:" << t1 << " t2:" << t2;
    // Shift eta1 back by split units in time 
    PiecewiseConstantRateFunction<T> shifted_eta1(shiftParams(params1, split), {t1 - split, t2 - split});
    const Matrix<T> rsfs = csfs.at(n1 + n2).compute(shifted_eta1)[0];
    Matrix<T> const& c_eMn2 = eMn2;
    std::array<Matrix<T>, 3> const& c_eMn1 = eMn1;

#pragma omp parallel for collapse(2)
    for (int b1 = 0; b1 < n1 + 1; ++b1)
    {
        for (int b2 = 0; b2 < n2 + 1; ++b2)
        {
            const T myweight = weight;
            for (int nseg = 0; nseg < n1 + n2 + 1; ++nseg)
            {
                for (int np1 = std::max(nseg - n2, 0); np1 < std::min(nseg, n1) + 1; ++np1)
                {
                    int np2 = nseg - np1;
                    // scipy.stats.hypergeom.pmf(np1, n1 + n2, nseg, n1)  = choose(nseg, np1) * choose(n1 + n2 - nseg, n1 - np1)
                    // gsl_ran_hypergeometric_pdf(unsigned int k, unsigned int n1, unsigned int n2, unsigned int t) 
                    double h = scipy_stats_hypergeom_pmf(np1, n1 + n2, nseg, n1);
                    for (int i = 0; i < 3; ++i)
                    {
                        T x = rsfs(i, nseg);
                        x *= c_eMn1[i](np1, b1);
                        x *= c_eMn2(np2, b2);
                        x *= h;
                        x *= myweight;
                        tensorRef(m, i, b1, 0, b2) += x;
                    }
                }
            }
        }
    }
     
    // pop 1, below split
    Matrix<T> sfs_below = csfs.at(n1).compute_below(*eta1)[0];
    for (int i = 0; i < a1 + 1; ++i)
        for (int j = 0; j < n1 + 1; ++j)
        {
            assert(sfs_below(i, j) > -1e-8);
            if (sfs_below(i, j) > 0)
                tensorRef(m, i, j, 0, 0) += weight * sfs_below(i, j);
        }
}

template <typename T>
std::vector<Matrix<T> > JointCSFS<T>::compute(const PiecewiseConstantRateFunction<T>&)
{
    if (a1 == 1 and a2 == 1)
        pre_compute_apart();
    else if (a1 == 2 and a2 == 0)
        pre_compute_together();
    else
        throw std::runtime_error("unsupported jcsfs configuration");
    for (int m = 0; m < M; ++m)
    {
        // Threshold jcsfs to have minimum value.
        J[m] = J[m].unaryExpr([] (T x)
                {
                    if (x > 1e-20)
                        return x;
                    x *= 0.;
                    x += 1e-20;
                    return x;
                });
        // zero out nonsegregating sites
        tensorRef(m, 0, 0, 0, 0) *= 0.;
        tensorRef(m, a1, n1, a2, n2) *= 0;
        CHECK_NAN(J[m]);
    }
    return J;
}


// Public class methods
template <typename T>
void JointCSFS<T>::pre_compute(const ParameterVector &params1, 
        const ParameterVector &params2, double split)
{
    this->split = split;
    this->params1 = params1;
    this->params2 = params2;
}

template <typename T>
void JointCSFS<T>::pre_compute_apart()
{
    // Compute the JCSFS in the case where both distinguished lineages
    // are different subpopulations;
    
    // Above the split its the usual CSFS + moran, with 
    // some additional combinatorial stuff 
    // Undistinguished SFS beneath split for each subpop, times
    // combinatorial fac that the lineage contains the d.l.
    std::vector<double> times{0};
    for (int m = 1; m < M; ++m)
    {
        double t1 = hidden_states[m];
        if (t1 > split)
            times.push_back(t1 - split);
    }
    times.push_back(INFINITY);
    const PiecewiseConstantRateFunction<T> shifted_eta1(shiftParams(params1, split), times);
    const std::vector<Matrix<T> > csfs_at_split = csfs.at(n1 + n2).compute(shifted_eta1);
    const T Rts1 = PiecewiseConstantRateFunction<T>(params1, {}).R(split);
    const T Rts2 = PiecewiseConstantRateFunction<T>(params2, {}).R(split);
    const Matrix<T> T10 = apartM.Mn10.expM(Rts1);
    const Matrix<T> T11 = apartM.Mn11.expM(Rts1);
    const Matrix<T> T20 = apartM.Mn20.expM(Rts2);
    const Matrix<T> T21 = apartM.Mn21.expM(Rts2);
    int i = 0;
    for (int m = 0; m < M; ++m)
    {
        J[m].fill(shifted_eta1.zero());
        // Under this model there a1 and a2 cannot coalesce beneath
        // split. So we don't bother calculating the emission
        // distribution conditional on this null event.
        const double t2 = hidden_states[m + 1];
        if (t2 <= split)
            continue;
        const Matrix<T> csfs_shift = csfs_at_split[i++];
#pragma omp parallel for collapse(2)
        for (int b1 = 0; b1 <= n1; ++b1)
        {
            for (int b2 = 0; b2 <= n2; ++b2)
            {
                for (int nseg = 0; nseg <= n1 + n2; ++nseg)
                {
                    for (int np1 = std::max(nseg - n2, 0); np1 <= std::min(nseg, n1); ++np1)
                    {
                        const int np2 = nseg - np1;
                        const double h = scipy_stats_hypergeom_pmf(np1, n1 + n2, nseg, n1);
                        tensorRef(m, 1, b1, 1, b2) += h * csfs_shift(2, nseg) * T11(np1, b1) * T21(np2, b2);
                        tensorRef(m, 1, b1, 0, b2) += 0.5 * h * csfs_shift(1, nseg) * T11(np1, b1) * T20(np2, b2);
                        tensorRef(m, 0, b1, 1, b2) += 0.5 * h * csfs_shift(1, nseg) * T10(np1, b1) * T21(np2, b2);
                        tensorRef(m, 0, b1, 0, b2) += h * csfs_shift(0, nseg) * T10(np1, b1) * T20(np2, b2);
                    }
                }
            }
        }
    }
    // Cover edge case
    if (split == 0.)
        return; 

    // Truncated SFS until split.
    auto t1 = std::make_tuple(params1, n1);
    auto t2 = std::make_tuple(params2, n2);
    std::vector<decltype(t1)> tups{t1, t2};
    bool first = true;
    for (auto t : tups)
    {
        ParameterVector trunc_params = truncateParams(std::get<0>(t), split);
        const int ni = std::get<1>(t);
        PiecewiseConstantRateFunction<T> eta_trunc(trunc_params, {0., INFINITY});
        Vector<T> rsfs_below;
        if (ni > 0)
            rsfs_below = undistinguishedSFS(csfs.at(ni - 1).compute(eta_trunc)[0]);
        for (int k = 1; k <= ni; ++k)
        {
            double fac = (double)k / (double)(ni + 1);
            const T x1 = (1. - fac) * rsfs_below(k - 1);
            const T x2 = fac * rsfs_below(k - 1);
            for (int m = 0; m < M; ++m)
            // a1 and a2 cannot coalesce beneath the split so this part is
            // the same for all hidden states.
            {
                if (first)
                {
                    tensorRef(m, 0, k, 0, 0) += x1;
                    tensorRef(m, 1, k - 1, 0, 0) += x2;
                }
                else
                {
                    tensorRef(m, 0, 0, 0, k) += x1;
                    tensorRef(m, 0, 0, 1, k - 1) += x2;
                }
            }
        }
        T remain = eta_trunc.zero();
        if (ni > 0)
            remain = arange(1, ni + 1).transpose().template cast<T>() * rsfs_below;
        remain /= (double)(ni + 1);
        remain -= split;
        assert(remain <= 0);
        for (int m = 0; m < M; ++m)
        {
            if (first)
                tensorRef(m, 1, ni, 0, 0) -= remain;
            else
                tensorRef(m, 0, 0, 1, ni) -= remain;
        }
        first = false;
    }
}


template <typename T>
void JointCSFS<T>::pre_compute_together()
{
    // Compute the JCSFS in the case where both distinguished lineages
    // are in the same subpopulation.
    eta1.reset(new PiecewiseConstantRateFunction<T>(params1, {split - 1e-6, split + 1e-6}));
    eta2.reset(new PiecewiseConstantRateFunction<T>(params2, {}));
    Rts1 = eta1->R(split);
    Rts2 = eta2->R(split);
    eMn1[0] = togetherM.Mn10.expM(Rts1);
    eMn1[1] = togetherM.Mn11.expM(Rts1);
    eMn1[2] = eMn1[0].reverse();
    eMn2 = togetherM.Mn2.expM(Rts2);
    for (int m = 0; m < M; ++m)
    {
        J[m].fill(eta1->zero());
        const double t1 = hidden_states[m], t2 = hidden_states[m + 1];
        if (t1 < t2 and t2 <= split)
            jcsfs_helper_tau_below_split(m, t1, t2, 1.); 
        else if (split <= t1 and t1 < t2)
            jcsfs_helper_tau_above_split(m, t1, t2, 1.); 
        else
        {
            T eR1t1 = exp(-eta1->R(t1));
            T eR1t2;
            if (std::isinf(t2))
                eR1t2 = eta1->zero();
            else
                eR1t2 = exp(-eta1->R(t2));
            T w = (exp(-Rts1) - eR1t2) / (eR1t1 - eR1t2);
            jcsfs_helper_tau_below_split(m, t1, split, 1. - w);
            jcsfs_helper_tau_above_split(m, split, t2, w);
        }
        // pop2, below split
        if (n2 == 1)
            tensorRef(m, 0, 0, 0, 1) += split;
        if (n2 > 1)
        {
            ParameterVector params2_trunc = truncateParams(params2, split);
            const PiecewiseConstantRateFunction<T> eta2_trunc(params2_trunc, {0., INFINITY});
            Vector<T> rsfs_below_2 = undistinguishedSFS(csfs.at(n2 - 2).compute(eta2_trunc)[0]);
            assert(rsfs_below_2.size() == n2 - 1);
            for (int i = 0; i < n2 - 1; ++i)
                tensorRef(m, 0, 0, 0, i + 1) += rsfs_below_2(i);
            const Vector<double> Sn2 = arange(1, n2) / n2;
            T remain = Sn2.transpose().template cast<T>() * rsfs_below_2;
            remain -= split;
            tensorRef(m, 0, 0, 0, n2) -= remain;
        }
    }
}

// Instantiate necessary templates
template class JointCSFS<double>;
template class JointCSFS<adouble>;

