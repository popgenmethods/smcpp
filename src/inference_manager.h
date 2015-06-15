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
    
    // Unfortunately these are necessary to work around a bug in Cython
    void setParams_d(const ParameterVector params) { setParams<double>(params); }
    void setParams_ad(const ParameterVector params) { setParams<adouble>(params); }

    private:
    // Passed-in parameters
    std::mt19937 gen;
    std::vector<std::vector<double>> params;
    const MatrixInterpolator moran_interp;
    const int n, L;
    const std::vector<int*> observations;
    const std::vector<double> hidden_states;
    double theta, rho;
    const int block_size, num_threads, num_samples, M;
    ThreadPool tp;

    std::vector<HMM> hmms;
    Vector<adouble> pi;
    Matrix<adouble> transition, emission;

    // Methods
    void parallel_do(std::function<void(HMM &)>);

    template <typename T>
    std::vector<T> parallel_select(std::function<T(HMM &)>);
};

#endif
