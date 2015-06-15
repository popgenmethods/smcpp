#ifndef CONDITIONED_SFS_H
#define CONDITIONED_SFS_H

#include <cassert>
#include <cfenv>
#include <map>
#include <thread>
#include <random>

#include "common.h"
#include "rate_function.h"
#include "piecewise_exponential_rate_function.h"
#include "spline_rate_function.h"
#include "matrix_interpolator.h"


template <typename T>
class ConditionedSFS
{
    public:
    ConditionedSFS(const PiecewiseExponentialRateFunction<T>, int, const MatrixInterpolator);
    void compute(int, T, T);
    std::thread compute_threaded(int, T, T);
    Matrix<T> matrix() const { return csfs; }
    static Matrix<T> calculate_sfs(const PiecewiseExponentialRateFunction<T> &eta, int n, int num_samples, 
            const MatrixInterpolator &moran_interp, double tau1, double tau2, int numthreads, double theta);

    private:
    // Methods
    void fill_matrices();
    void construct_ad_vars();
    void compute_ETnk_below(const Vector<T>&);
    double exp1();
    T exp1_conditional(T, T);
    double unif();
    double rand_exp();
    static Matrix<T> average_csfs(std::vector<ConditionedSFS<T>> &csfs, double theta);

    // Variables
    std::mt19937 gen;
    const PiecewiseExponentialRateFunction<T> eta;
    const int n;
    const MatrixInterpolator moran_interp;
    Matrix<T> D_subtend_above, D_not_subtend_above, D_subtend_below, 
        D_not_subtend_below, Wnbj, P_dist, P_undist, tK, 
        csfs, csfs_above, csfs_below, ETnk_below;
};

void set_seed(long long);
void store_sfs_results(const Matrix<double>&, double*);
void store_sfs_results(const Matrix<adouble>&, double*, double*);

// These methods are used for testing purposes only
void cython_calculate_sfs(const std::vector<std::vector<double>> &params,
        int n, int num_samples, const MatrixInterpolator &,
        double tau1, double tau2, int numthreads, double theta, 
        double* outsfs);
void cython_calculate_sfs_jac(const std::vector<std::vector<double>> &params,
        int n, int num_samples, const MatrixInterpolator &,
        double tau1, double tau2, int numthreads, double theta, 
        double* outsfs, double* outjac);

void init_eigen();

#endif
