#ifndef INFERENCE_MANAGER_H
#define INFERENCE_MANAGER_H

#include "gsl/gsl_randist.h"

#include "common.h"
#include "hmm.h"
#include "piecewise_exponential_rate_function.h"
#include "conditioned_sfs.h"
#include "transition.h"
#include "transition_bundle.h"
#include "inference_bundle.h"

typedef std::vector<std::vector<double>> ParameterVector;
class HMM;

class InferenceManager
{
    public:
    InferenceManager(
            const int, const std::vector<int>,
            const std::vector<int*>,
            const std::vector<double>,
            const double theta, const double rho);
    template <typename T>
    void setParams(const ParameterVector, const std::vector<std::pair<int, int> >);

    void Estep(void);
    std::vector<adouble> Q(double);
    std::vector<double> loglik(double);

    template <typename T>
    std::vector<Matrix<T> > sfs(const PiecewiseExponentialRateFunction<T>&);

    // Unfortunately these are necessary to work around a bug in Cython
    void setParams_d(const ParameterVector params);
    void setParams_ad(const ParameterVector params, const std::vector<std::pair<int, int>> derivatives);
    double R(const ParameterVector params, double t);
    double getRegularizer();
    bool debug, saveGamma;
    std::vector<double> hidden_states;
    std::vector<double> randomCoalTimes(const ParameterVector params, double fac, const int size);
    std::map<block_key, Vector<adouble> > block_probs;
    std::vector<Matrix<double>*> getXisums();
    std::vector<Matrix<double>*> getGammas();
    Matrix<adouble>& getPi();
    Matrix<adouble>& getTransition();
    Matrix<adouble>& getEmission();

    private:
    template <typename T> 
    ConditionedSFS<T>& getCsfs();
    std::vector<Eigen::Matrix<int, Eigen::Dynamic, 4, Eigen::RowMajor> > map_obs(const std::vector<int*>&, const std::vector<int>&);
    std::set<int> fill_spans();
    std::set<block_key> fill_targets();
    Matrix<double>& subEmissionCoefs(int);
    template <typename T>
    void recompute_B(const PiecewiseExponentialRateFunction<T> &);
    void populate_block_probs();
    typedef std::unique_ptr<HMM> hmmptr;

    // Passed-in parameters
    std::mt19937 gen;
    const int n;
    std::vector<Eigen::Matrix<int, Eigen::Dynamic, 4, Eigen::RowMajor> > obs;
    double theta, rho;
    const int M;
    adouble regularizer;

    std::vector<hmmptr> hmms;
    Vector<adouble> pi;
    Matrix<adouble> transition, emission;
    ConditionedSFS<double> csfs_d;
    ConditionedSFS<adouble> csfs_ad;
    std::vector<block_key> bpm_keys;
    std::set<int> nbs;
    std::map<int, Matrix<double> > subEmissionCoefs_memo;
    std::set<int> spans;
    std::set<block_key> targets;
    TransitionBundle tb;
    InferenceBundle ib;

    // Methods
    void parallel_do(std::function<void(hmmptr &)>);
    template <typename T>
    std::vector<T> parallel_select(std::function<T(hmmptr &)>);
};

template <typename T>
Matrix<T> sfs_cython(int n, const ParameterVector &p, double t1, double t2, double theta) 
{ 
    PiecewiseExponentialRateFunction<T> eta(p, {t1, t2});
    ConditionedSFS<T> csfs(n - 2, 1);
    return csfs.compute(eta, theta)[0];
}

template <typename T>
Matrix<T> sfs_cython(int n, const ParameterVector &p, double t1, double t2, double theta, std::vector<std::pair<int, int> > deriv) 
{ 
    PiecewiseExponentialRateFunction<T> eta(p, deriv, {t1, t2});
    ConditionedSFS<T> csfs(n - 2, 1);
    return csfs.compute(eta, theta)[0];
}

#endif
