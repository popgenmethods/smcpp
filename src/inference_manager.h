#ifndef INFERENCE_MANAGER_H
#define INFERENCE_MANAGER_H

#include "common.h"
#include "matrix_interpolator.h"
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
            const MatrixInterpolator &moran_interp,
            const int n, const int L,
            const std::vector<int*> observations,
            const std::vector<double> hidden_states,
            const double theta, const double rho, 
            const int block_size, const int num_threads,
            const int num_samples);
    
    template <typename T>
    void setParams(const ParameterVector);

    void Estep(void);
    std::vector<adouble> Q(double);
    std::vector<double> loglik(double);

    template <typename T>
    Matrix<T> sfs(const ParameterVector, double, double);
    Matrix<double> sfs_cython(const ParameterVector p, double t1, double t2) { return sfs<double>(p, t1, t2); }
    Matrix<adouble> dsfs_cython(const ParameterVector p, double t1, double t2) { return sfs<adouble>(p, t1, t2); }
    
    // Unfortunately these are necessary to work around a bug in Cython
    void setParams_d(const ParameterVector params) { setParams<double>(params); }
    void setParams_ad(const ParameterVector params) { setParams<adouble>(params); }

    double R(const ParameterVector params, double t)
    {
        PiecewiseExponentialRateFunction<double> eta(params);
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

    private:
    // Passed-in parameters
    std::mt19937 gen;
    const MatrixInterpolator moran_interp;
    const int n, L;
    const std::vector<int*> observations;
    const std::vector<double> hidden_states;
    double theta, rho;
    const int block_size, num_threads;
    int num_samples;
    const int M;
    ThreadPool tp;
    adouble regularizer;

    std::vector<HMM> hmms;
    Vector<adouble> pi;
    Matrix<adouble> transition, emission;

    // Methods
    void parallel_do(std::function<void(HMM &)>);

    template <typename T>
    std::vector<T> parallel_select(std::function<T(HMM &)>);
};

#endif
