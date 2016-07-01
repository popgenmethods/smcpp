#ifndef INFERENCE_MANAGER_H
#define INFERENCE_MANAGER_H

#include "common.h"
#include "hmm.h"
#include "piecewise_exponential_rate_function.h"
#include "conditioned_sfs.h"
#include "transition.h"
#include "transition_bundle.h"
#include "inference_bundle.h"

typedef std::vector<std::vector<adouble>> ParameterVector;
class HMM;

class InferenceManager
{
    public:
    InferenceManager(
            const int, const std::vector<int>, const std::vector<int*>,
            const std::vector<double>, const std::vector<double>);

    void setRho(const double);
    void setTheta(const double);
    void setParams(const ParameterVector);

    void Estep(bool);
    std::vector<adouble> Q();
    std::vector<double> loglik();

    template <typename T>
    std::vector<Matrix<T> > sfs(const PiecewiseExponentialRateFunction<T>&);

    // Unfortunately these are necessary to work around a bug in Cython
    double R(const ParameterVector params, double t);
    bool debug, saveGamma, folded;
    std::vector<double> hidden_states, s;
    std::map<block_key, Vector<adouble> > emission_probs;
    std::vector<Matrix<double>*> getXisums();
    std::vector<Matrix<double>*> getGammas();
    std::vector<std::map<block_key, Vector<double> >*> getGammaSums();
    Matrix<adouble>& getPi();
    Matrix<adouble>& getTransition();
    Matrix<adouble>& getEmission();
    std::map<block_key, Vector<adouble> >& getEmissionProbs();

    private:
    ParameterVector params;
    std::vector<std::pair<int, int> > derivatives;
    double theta, rho;

    std::vector<Eigen::Matrix<int, Eigen::Dynamic, 4, Eigen::RowMajor> > 
        map_obs(const std::vector<int*>&, const std::vector<int>&);
    std::set<std::pair<int, block_key> > fill_targets();
    std::vector<int> fill_nbs();
    std::map<int, Matrix<double> > fill_subemissions();
    void populate_emission_probs();
    void recompute_initial_distribution();
    void recompute_emission_probs();
    typedef std::unique_ptr<HMM> hmmptr;

    // Passed-in parameters
    std::mt19937 gen;
    const int n;
    std::vector<Eigen::Matrix<int, Eigen::Dynamic, 4, Eigen::RowMajor> > obs;
    const int M;
    ConditionedSFS<adouble> csfs;

    std::vector<hmmptr> hmms;
    Vector<adouble> pi;
    Matrix<adouble> transition, emission;
    std::vector<block_key> bpm_keys;
    const std::vector<int> nbs;
    const std::map<int, Matrix<double> > subEmissionCoeffs;
    const std::set<std::pair<int, block_key> > targets;
    TransitionBundle tb;
    std::vector<Matrix<adouble> > sfss;

    public:
    int spanCutoff;

    private:
    InferenceBundle ib;
    struct { bool theta, rho, params; } dirty;

    // Methods
    void parallel_do(std::function<void(hmmptr &)>);
    template <typename T>
    std::vector<T> parallel_select(std::function<T(hmmptr &)>);
    void do_dirty_work();
    PiecewiseExponentialRateFunction<adouble> getEta();
};

Matrix<adouble> sfs_cython(int n, const ParameterVector &p, std::vector<double> s,
        double t1, double t2);

#endif
