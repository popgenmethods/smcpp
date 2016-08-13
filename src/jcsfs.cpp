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
                tensorRef(m, i, j, 0, 0) = weight * trunc_csfs(i, j);
        }
    const Vector<T> trunc_sfs = undistinguishedSFS(trunc_csfs);
    T Et = Sn1.transpose().template cast<T>() * trunc_sfs;
    tensorRef(m, 2, n1, 0, 0) = split - Et;
    tensorRef(m, 2, n1, 0, 0) *= weight;

    // Above split, then moran down
    const ParameterVector params1_shift = shiftParams(params1, split);
    const PiecewiseConstantRateFunction<T> eta1_shift(params1_shift, {0., INFINITY});
    Vector<T> sfs_above_split = undistinguishedSFS(csfs.at(n1 + n2 - 1).compute(eta1_shift)[0]);
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
                for (int np1 = std::max(nseg - n2, 0); np1 < std::min(nseg, n1 + 1) + 1; ++np1)
                {
                    int np2 = nseg - np1;
                    double h = scipy_stats_hypergeom_pmf(np1, n1 + n2 + 1, nseg, n1 + 1);
                    T x = sfs_above_split(nseg - 1) * eMn2(np2, b2);
                    x *= h;
                    x *= weight;
                    T x0 = x * eMn10_avg(np1, b1);
                    T x2 = x * eMn12_avg(np1, b1);
                    tensorRef(m, 0, b1, 0, b2) += x0;
                    tensorRef(m, 2, b1, 0, b2) += x2;
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
                for (int np1 = std::max(nseg - n2, 0); np1 < std::min(nseg, n1) + 1; ++np1)
                {
                    int np2 = nseg - np1;
                    // scipy.stats.hypergeom.pmf(np1, n1 + n2, nseg, n1)  = choose(nseg, np1) * choose(n1 + n2 - nseg, n1 - np1)
                    // gsl_ran_hypergeometric_pdf(unsigned int k, unsigned int n1, unsigned int n2, unsigned int t) 
                    double h = scipy_stats_hypergeom_pmf(np1, n1 + n2, nseg, n1);
                    for (int i = 0; i < 3; ++i)
                    {
                        T x = rsfs(i, nseg) * eMn1[i](np1, b1) * eMn2(np2, b2);
                        x *= h;
                        x *= weight;
                        tensorRef(m, i, b1, 0, b2) += x;
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
    if (a1 == 1 and a2 == 1)
        pre_compute_apart();
    else if (a1 == 2 and a2 == 0)
        pre_compute_together();
    else
        throw std::runtime_error("unsupported jcsfs configuration");
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
    PiecewiseConstantRateFunction<T> shifted_eta1(shiftParams(params1, split), times);
    std::vector<Matrix<T> > csfs_at_split = csfs.at(n1 + n2).compute(shifted_eta1);
    T Rts1 = PiecewiseConstantRateFunction<T>(params1, {}).R(split);
    T Rts2 = PiecewiseConstantRateFunction<T>(params2, {}).R(split);
    Matrix<T> T10 = Mn10.expM(Rts1);
    Matrix<T> T20 = Mn20.expM(Rts2);
    Matrix<T> T11 = Mn11.expM(Rts1);
    Matrix<T> T21 = Mn21.expM(Rts2);
    int i = 0;
    for (int m = 0; m < M; ++m)
    {
        J[m].setZero();
        // Under this model there a1 and a2 cannot coalesce beneath
        // split. So we don't bother calculating the emission
        // distribution conditional on this null event.
        double t2 = hidden_states[m + 1];
        if (t2 <= split)
            continue;
        Matrix<T> csfs_shift = csfs_at_split[i++];
        for (int b1 = 0; b1 < n1 + 1; ++b1)
            for (int b2 = 0; b2 < n2 + 1; ++b2)
                for (int nseg = 0; nseg < n1 + n2 + 1; ++nseg)
                    for (int np1 = std::max(nseg - n2, 0); np1 < std::min(nseg, n1) + 1; ++np1)
                    {
                        int np2 = nseg - np1;
                        double h = scipy_stats_hypergeom_pmf(np1, n1 + n2, nseg, n1);
                        tensorRef(m, 1, b1, 1, b2) += h * csfs_shift(2, nseg) * T11(np1, b1) * T21(np2, b2);
                        tensorRef(m, 0, b1, 1, b2) += 0.5 * h * csfs_shift(1, nseg) * T10(np1, b1) * T21(np2, b2);
                        tensorRef(m, 1, b1, 0, b2) += 0.5 * h * csfs_shift(1, nseg) * T11(np1, b1) * T20(np2, b2);
                        tensorRef(m, 0, b1, 0, b2) += h * csfs_shift(0, nseg) * T10(np1, b1) * T20(np2, b2);
                    }
    }
    // Cover edge case.
    if (split == 0.)
        return; 

    // Truncated SFS until split.
    auto t1 = std::make_tuple(truncateParams(params1, split), n1);
    auto t2 = std::make_tuple(truncateParams(params2, split), n2);
    std::vector<decltype(t1)> tups{t1, t2};
    bool first = true;
    for (auto t : tups)
    {
        int ni = std::get<1>(t);
        PiecewiseConstantRateFunction<T> eta_trunc(std::get<0>(t), {0., INFINITY});
        Vector<T> rsfs_below = undistinguishedSFS(csfs.at(ni - 1).compute(eta_trunc)[0]);
        for (int k = 1; k < ni + 1; ++k)
        {
            double fac = (double)k / double(ni + 1);
            T x1 = (1. - fac) * rsfs_below(k - 1);
            T x2 = fac * rsfs_below(k - 1);
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
        T remain = arange(1, ni + 1).transpose().template cast<T>() * rsfs_below;
        remain /= ni + 1;
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
            T eR1t1 = exp(-eta1->R(t1));
            T eR1t2 = exp(-eta1->R(t2));
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
            T remain = Sn2.transpose().template cast<T>() * rsfs_below_2;
            remain -= split;
            tensorRef(m, 0, 0, 0, n2) -= remain;
        }
        // zero out nonsegregating sites 
        tensorRef(m, 0, 0, 0, 0) *= 0.;
        tensorRef(m, a1, n1, 0, n2) *= 0;
    }
}

// Instantiate necessary templates
template class JointCSFS<double>;
template class JointCSFS<adouble>;

