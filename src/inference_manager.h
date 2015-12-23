#ifndef INFERENCE_MANAGER_H
#define INFERENCE_MANAGER_H

#include "gsl/gsl_randist.h"

#include "common.h"
#include "hmm.h"
#include "piecewise_exponential_rate_function.h"
#include "conditioned_sfs.h"
#include "transition.h"
#include "block_key.h"

typedef std::vector<std::vector<double>> ParameterVector;

class HMM;

class InferenceManager
{
    public:
    InferenceManager(
            const int n, const std::vector<int> L,
            const std::vector<int*> observations,
            const std::vector<double> hidden_states,
            const int* emission_mask,
            const int mask_freq,
            const double theta, const double rho, 
            const int block_size);
    
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
    bool debug, hj, forwardOnly, saveGamma;
    std::vector<double> hidden_states;
    std::vector<double> randomCoalTimes(const ParameterVector params, double fac, const int size);
    std::unordered_map<block_key, Vector<adouble> > block_prob_map;
    std::vector<Matrix<float>*> getXisums();
    std::vector<Matrix<float>*> getGammas();
    std::vector<Matrix<adouble>*> getBs();
    std::vector<block_key_vector> getBlockKeys();
    Matrix<adouble>& getPi();
    Matrix<adouble>& getTransition();
    Matrix<adouble>& getEmission();

    private:
    template <typename T> 
    ConditionedSFS<T>& getCsfs();
    Matrix<double>& subEmissionCoefs(int);
    void recompute_B();
    void populate_block_prob_map();
    typedef std::unique_ptr<HMM> hmmptr;

    // Passed-in parameters
    std::mt19937 gen;
    const int n;
    const std::vector<int> L;
    const std::vector<int*> observations;
    const Eigen::Matrix<int, 3, Eigen::Dynamic, Eigen::RowMajor> emask, two_mask;
    const int mask_freq;
    double theta, rho;
    const int block_size;
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
