#ifndef INFERENCE_MANAGER_H
#define INFERENCE_MANAGER_H

#include <vector>
#include <map>
#include <utility>
#include <memory>

#include "common.h"
#include "transition_bundle.h"
#include "inference_bundle.h"
#include "piecewise_constant_rate_function.h"
#include "conditioned_sfs.h"
#include "hmm.h"
#include "block_key.h"

class InferenceManager
{
    public:
    InferenceManager(
            const int,
            const int,
            const std::vector<int>, 
            const std::vector<int*>,
            const std::vector<double>,
            ConditionedSFS<adouble> *csfs);
    virtual ~InferenceManager() = default;

    void setRho(const double);
    void setTheta(const double);
    void setAlpha(const double);

    void Estep(bool);
    std::vector<adouble> Q();
    std::vector<double> loglik();

    void setParams(const ParameterVector &params);

    bool saveGamma;
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

    // Methods
    void parallel_do(std::function<void(hmmptr &)>);
    template <typename T> std::vector<T> parallel_select(std::function<T(hmmptr &)>);
    void recompute_initial_distribution();
    std::vector<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > 
        map_obs(const std::vector<int*>&, const std::vector<int>&);
    std::set<std::pair<int, block_key> > fill_targets();
    void populate_emission_probs();
    void do_dirty_work();

    // These methods will differ according to number of populations and must be overridden.
    virtual void recompute_emission_probs() = 0;

    // Other members
    const int npop, sfs_dim, M;
    std::vector<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > obs;
    std::unique_ptr<ConditionedSFS<adouble> > csfs;
    double theta, rho, alpha;
    std::vector<hmmptr> hmms;
    Vector<adouble> pi;
    Matrix<adouble> transition, emission;
    std::vector<block_key> bpm_keys;
    const std::set<std::pair<int, block_key> > targets;
    TransitionBundle tb;
    std::vector<Matrix<adouble> > sfss;

    InferenceBundle ib;
    struct { bool theta, rho, eta; } dirty;

    std::unique_ptr<const PiecewiseConstantRateFunction<adouble> > eta;
};

template <size_t P>
class NPopInferenceManager : public InferenceManager
{
    public:
    NPopInferenceManager(
            const FixedVector<int, P> n,
            const FixedVector<int, P> na,
            const std::vector<int> obs_lengths,
            const std::vector<int*> observations,
            const std::vector<double> hidden_states,
            ConditionedSFS<adouble> *csfs,
            const double polarization_error) :
        InferenceManager(P,
                (na.tail(na.size() - 1).array() + 1).prod() * (n.array() + 1).prod(),
                obs_lengths, observations, hidden_states, csfs),
                n(n), na(na), tensordims(make_tensordims()),
                bins(construct_bins(polarization_error))
    {}

    virtual ~NPopInferenceManager() = default;


    protected:
    // Virtual overrides
    void recompute_emission_probs();
    bool is_monomorphic(const block_key&);
    block_key folded_key(const block_key&);
    block_key_prob_map merge_monomorphic(const block_key_prob_map&);
    FixedVector<int, 2 * P> make_tensordims();
    block_key bk_to_map_key(const block_key &bk);

    Vector<adouble> tensorRef(const block_key &bk);

    // Passed-in parameters
    const FixedVector<int, P> n;
    const FixedVector<int, P> na;
    const FixedVector<int, 2 * P> tensordims;

    std::map<block_key, std::map<block_key, double> > construct_bins(const double);
    std::map<block_key, std::map<block_key, double> > bins;
};

class OnePopInferenceManager final : public NPopInferenceManager<1>
{
    public:
    OnePopInferenceManager(
            const int n,
            const std::vector<int> obs_lengths,
            const std::vector<int*> observations,
            const std::vector<double> hidden_states,
            const double);
};

class TwoPopInferenceManager : public NPopInferenceManager<2>
{
    public:
    TwoPopInferenceManager(
            const int n1, const int n2,
            const int a1, const int a2,
            const std::vector<int> obs_lengths,
            const std::vector<int*> observations,
            const std::vector<double> hidden_states,
            const double);
                
    void setParams(const ParameterVector&, const ParameterVector&, const ParameterVector&, const double);

    private:
    const int a1, a2;
};

Matrix<adouble> sfs_cython(const int, const ParameterVector, const double, const double, bool);

#endif
