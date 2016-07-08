#ifndef INFERENCE_MANAGER_H
#define INFERENCE_MANAGER_H

#include "common.h"
#include "hmm.h"
#include "piecewise_exponential_rate_function.h"
#include "conditioned_sfs.h"
#include "transition.h"
#include "transition_bundle.h"
#include "inference_bundle.h"
#include "demography.h"

typedef std::vector<std::vector<adouble>> ParameterVector;
class HMM;

template <int P>
class InferenceManager
{
    public:
    InferenceManager(
            const Demography<P>,
            const std::vector<int>, 
            const std::vector<int*>,
            const std::vector<double>);

    void setRho(const double);
    void setTheta(const double);
    void setParams(const std::array<ParameterVector, P>);

    void Estep(bool);
    std::vector<adouble> Q();
    std::vector<double> loglik();

    template <typename T>
    std::vector<Matrix<T> > sfs(const PiecewiseExponentialRateFunction<T>&);

    bool saveGamma, folded;
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
    std::array<ParameterVector, P> params;
    double theta, rho;

    std::vector<Eigen::Matrix<int, Eigen::Dynamic, 2 + 2 * P, Eigen::RowMajor> > 
        map_obs(const std::vector<int*>&, const std::vector<int>&);
    std::set<std::pair<int, block_key<P> > > fill_targets();
    void populate_emission_probs();
    void recompute_initial_distribution();
    void recompute_emission_probs();
    typedef std::unique_ptr<HMM> hmmptr;

    // Passed-in parameters
    std::mt19937 gen;
    const int n;
    std::vector<Eigen::Matrix<int, Eigen::Dynamic, 2 + 2 * P, Eigen::RowMajor> > obs;
    const int M;
    ConditionedSFSBase<adouble, P> csfs;

    std::vector<hmmptr> hmms;
    Vector<adouble> pi;
    Matrix<adouble> transition, emission;
    std::vector<block_key<P> > bpm_keys;
    const std::set<std::pair<int, block_key<P> > > targets;
    TransitionBundle tb;
    std::vector<Matrix<adouble> > sfss;

    protected:
    InferenceBundle ib;
    struct { bool theta, rho, params; } dirty;

    // Methods
    void parallel_do(std::function<void(hmmptr &)>);
    template <typename T>
    std::vector<T> parallel_select(std::function<T(hmmptr &)>);
    void do_dirty_work();
    Demography<adouble, P> getDemography();
};

Matrix<adouble> sfs_cython(const int, const ParameterVector, const std::vector<double>, const double, const double, bool);

#endif
