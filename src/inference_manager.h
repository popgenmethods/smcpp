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
    void Estep(void);

    template <typename T>
    void setParams(const ParameterVector);

    template <typename T>
    std::vector<T> Q(double);

    template <typename T>
    std::vector<T> loglik(double);

    template <typename T>
    Matrix<T> sfs(const ParameterVector, double, double);

    // Unfortunately these are necessary to work around a bug in Cython
    void setParams_d(const ParameterVector params) { setParams<double>(params); }
    void setParams_ad(const ParameterVector params) { setParams<adouble>(params); }
    void Estep_d(void) { Estep<double>(); }
    void Estep_ad(void) { Estep<adouble>(); }
    std::vector<double> Q_d(double lambda) { return Q<double>(lambda); }
    std::vector<adouble> Q_ad(double lambda) { return Q<adouble>(lambda); }
    std::vector<double> loglik_d(double lambda) { return loglik<double>(lambda); }
    std::vector<adouble> loglik_ad(double lambda) { return loglik<adouble>(lambda); }
    Matrix<double> sfs_d(const ParameterVector p, double t1, double t2) { return sfs<double>(p, t1, t2); }

    private:
    // Passed-in parameters
    std::vector<std::vector<double>> params;
    const MatrixInterpolator moran_interp;
    const int n, L;
    const std::vector<int*> observations;
    const std::vector<double> hidden_states;
    double theta, rho;
    const int block_size, num_threads, num_samples, M;
    ThreadPool tp;

    template <typename T>
    struct HMMBundle
    {
        std::vector<HMM<T>> hmms;
        Vector<T> pi;
        Matrix<T> transition, emission;
    };

    // Constructed variables
    HMMBundle<double> d_bundle;
    HMMBundle<adouble> ad_bundle;

    // Methods
    template <typename T>
    void parallel_do(std::function<void(HMM<T> &)>);

    template <typename T>
    std::vector<T> parallel_select(std::function<T(HMM<T> &)>);

    template <typename T>
    HMMBundle<T>& getBundle();
};

#endif
