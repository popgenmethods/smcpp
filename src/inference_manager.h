#ifndef INFERENCE_MANAGER_H
#define INFERENCE_MANAGER_H

#include <memory>

#include "common.h"
#include "ThreadPool.h"
#include "hmm.h"
#include "piecewise_exponential_rate_function.h"
#include "conditioned_sfs.h"
#include "transition.h"

typedef std::vector<std::vector<double>> ParameterVector;

class InferenceManager
{
    public:
    InferenceManager(
            const int n, const int L,
            const std::vector<int*> observations,
            const std::vector<double> hidden_states,
            const int* emission_mask,
            const int mask_freq,
            const std::vector<int> mask_offset,
            const double theta, const double rho, 
            const int block_size, const int num_threads,
            const int num_samples);
    
    void set_seed(long long s) { seed = s; }

    template <typename T>
    void setParams(const ParameterVector, const std::vector<std::pair<int, int>>);

    void Estep(void);
    std::vector<adouble> Q(double);
    std::vector<double> loglik(double);

    template <typename T>
    std::vector<Matrix<T> > sfs(const PiecewiseExponentialRateFunction<T>&);

    Matrix<double> sfs_cython(const ParameterVector p, double t1, double t2) 
    { 
        PiecewiseExponentialRateFunction<double> eta(p, {t1, t2});
        return sfs<double>(eta)[0];
    }
    Matrix<adouble> dsfs_cython(const ParameterVector p, double t1, double t2) 
    { 
        PiecewiseExponentialRateFunction<adouble> eta(p, {t1, t2});
        return sfs<adouble>(eta)[0];
    }
    
    // Unfortunately these are necessary to work around a bug in Cython
    void setParams_d(const ParameterVector params) 
    { 
        std::vector<std::pair<int, int>> d;
        setParams<double>(params, d);
    }
    void setParams_ad(const ParameterVector params, 
            const std::vector<std::pair<int, int>> derivatives) 
    {  
        setParams<adouble>(params, derivatives);
    }

    double R(const ParameterVector params, double t)
    {
        PiecewiseExponentialRateFunction<double> eta(params, std::vector<std::pair<int, int> >(), std::vector<double>());
        return (*eta.getR())(t);
    }

    void set_num_samples(int nsamples) { num_samples = nsamples; }

    bool debug;
    std::vector<Matrix<double>*> getAlphas();
    std::vector<Matrix<double>*> getBetas();
    std::vector<Matrix<double>*> getGammas();
    std::vector<Matrix<adouble>*> getBs();
    Matrix<double> getPi();
    Matrix<double> getTransition();
    Matrix<double> getEmission();
    Matrix<double> getMaskedEmission();

    private:
    template <typename T> 
    ConditionedSFS<T>& getCsfs();

    typedef std::unique_ptr<HMM> hmmptr;

    // Passed-in parameters
    std::mt19937 gen;
    const int n, L;
    const std::vector<int*> observations;
    const std::vector<double> hidden_states;
    const Eigen::Map<const Eigen::Matrix<int, 3, Eigen::Dynamic, Eigen::RowMajor>> emask;
    const int mask_freq;
    const std::vector<int> mask_offset;
    double theta, rho;
    const int block_size, num_threads;
    int num_samples;
    const int M;
    ThreadPool tp;
    long long seed;
    adouble regularizer;

    std::vector<hmmptr> hmms;
    Vector<adouble> pi;
    Matrix<adouble> transition, emission, emission_mask;
    ConditionedSFS<double> csfs_d;
    ConditionedSFS<adouble> csfs_ad;

    // Methods
    void parallel_do(std::function<void(hmmptr &)>);

    template <typename T>
    std::vector<T> parallel_select(std::function<T(hmmptr &)>);
};

#endif
