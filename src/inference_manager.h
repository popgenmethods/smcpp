#ifndef INFERENCE_MANAGER_H
#define INFERENCE_MANAGER_H

#include "common.h"
#include "hmm.h"
#include "piecewise_constant_rate_function.h"
#include "conditioned_sfs.h"
#include "transition.h"
#include "transition_bundle.h"
#include "inference_bundle.h"
#include "demography.h"
#include "block_key.h"
#include "ndarray.h"

class HMM;

class InferenceManager
{
    public:
    InferenceManager(
            const int,
            const std::vector<int>, 
            const std::vector<int*>,
            const std::vector<double>,
            const ConditionedSFS<adouble> csfs);

    void setRho(const double);
    void setTheta(const double);

    void Estep(bool);
    std::vector<adouble> Q();
    std::vector<double> loglik();

    template <typename T>
    std::vector<Matrix<T> > sfs(const PiecewiseConstantRateFunction<T>&);

    bool saveGamma, folded;
    std::vector<double> hidden_states;
    std::map<block_key, Vector<adouble> > emission_probs;
    std::vector<Matrix<double>*> getXisums();
    std::vector<Matrix<double>*> getGammas();
    std::vector<std::map<block_key, Vector<double> >*> getGammaSums();
    Matrix<adouble>& getPi();
    Matrix<adouble>& getTransition();
    Matrix<adouble>& getEmission();
    std::map<block_key, Vector<adouble> >& getEmissionProbs();

    protected:
    typedef std::unique_ptr<HMM> hmmptr;

    virtual PiecewiseConstantRateFunction<adouble> getDistinguishedEta();
    virtual std::vector<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > 
        map_obs(const std::vector<int*>&, const std::vector<int>&);
    virtual std::set<std::pair<int, block_key> > fill_targets();
    virtual void populate_emission_probs();
    virtual void recompute_emission_probs();
    virtual void do_dirty_work();

    // Methods
    void parallel_do(std::function<void(hmmptr &)>);
    template <typename T> std::vector<T> parallel_select(std::function<T(hmmptr &)>);
    void recompute_initial_distribution();

    // Passed-in parameters
    std::vector<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > obs;
    const int M, n_undistinguished;
    const ConditionedSFS<adouble> csfs;

    double theta, rho;
    std::vector<hmmptr> hmms;
    Vector<adouble> pi;
    Matrix<adouble> transition, emission;
    std::vector<block_key> bpm_keys;
    const std::set<std::pair<int, block_key> > targets;
    TransitionBundle tb;
    std::vector<Matrix<adouble> > sfss;

    InferenceBundle ib;
    struct { bool theta, rho, demo; } dirty;
};

template <size_t P>
class NPopInferenceManager : public InferenceManager
{
    public:
    NPopInferenceManager(
            const Eigen::Array<int, P, 1> n, 
            const std::vector<int> obs_lengths,
            const std::vector<int*> observations,
            const std::vector<double> hidden_states,
            const ConditionedSFS<adouble> csfs) :
        InferenceManager((n + 1).prod(), obs_lengths, observations, hidden_states, csfs), n(n) {}

    protected:
    // Virtual overrides
    PiecewiseConstantRateFunction<adouble> getDistinguishedEta();
    std::vector<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > 
        map_obs(const std::vector<int*>&, const std::vector<int>&);
    std::set<std::pair<int, block_key> > fill_targets();
    void populate_emission_probs();
    void recompute_emission_probs();
    void do_dirty_work();

    // Other methods
    void setDemography(const Demography<adouble, P>);

    // Passed-in parameters
    const Eigen::Array<int, P, 1> n;

    // Other parameters
    Demography<adouble, P> demo;
};

class OnePopInferenceManager final : public NPopInferenceManager<1>
{
    public:
    OnePopInferenceManager(
            const int n,
            const std::vector<int> obs_lengths,
            const std::vector<int*> observations,
            const std::vector<double> hidden_states) :
        NPopInferenceManager((Eigen::Array<int, 1, 1>() << n).finished(), 
                obs_lengths, observations, hidden_states, 
                OnePopConditionedSFS<adouble>(n, hidden_states.size() - 1)) {}
    void setParams(const ParameterVector params)
    {
        setDemography(OnePopDemography<adouble>(params, hidden_states));
    }
};

class TwoPopInferenceManager : NPopInferenceManager<2>
{

};

Matrix<adouble> sfs_cython(const int, const ParameterVector, const std::vector<double>, const double, const double, bool);

#endif
