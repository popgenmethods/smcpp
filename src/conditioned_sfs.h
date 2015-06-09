#ifndef CONDITIONED_SFS_H
#define CONDITIONED_SFS_H

#include <cassert>
#include <cfenv>
#include <map>
#include <thread>

#include "common.h"
#include "rate_function.h"
#include "piecewise_exponential_rate_function.h"
#include "spline_rate_function.h"

#define EIGEN_NO_AUTOMATIC_RESIZING 1

template <typename T>
class ConditionedSFS
{
    public:
    ConditionedSFS(const RateFunction<T>&, int);
    void compute(int, int, const std::vector<double>&, 
            const std::vector<Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>> &,
            T, T);
    std::thread compute_threaded(int, int, const std::vector<double>&, 
            const std::vector<Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>> &,
            T, T);
    Matrix<T> matrix() const;
    static Vector<T> calculate_sfs(const RateFunction<T> &eta, int n, int S, int M, const std::vector<double> &ts, 
            const std::vector<double*> &expM, double tau1, double tau2, int numthreads, double theta);

    private:
    // Methods
    void fill_matrices();
    void construct_ad_vars();
    void compute_ETnk_below(const Vector<T>&);
    double exp1();
    T exp1_conditional(T, T);
    double unif();
    static Matrix<T> average_csfs(std::vector<ConditionedSFS<T>> &csfs, double theta);

    // Variables
    std::mt19937 gen;
    const RateFunction<T> *eta;
    const int n;
    Matrix<T> D_subtend_above, D_not_subtend_above, D_subtend_below, 
        D_not_subtend_below, Wnbj, P_dist, P_undist, tK, 
        csfs, csfs_above, csfs_below, ETnk_below;
};

void set_seed(long long);
void store_sfs_results(const Matrix<double>&, double*);
void store_sfs_results(const Matrix<adouble>&, double*, double*);

// These methods are used for testing purposes only
void cython_calculate_sfs(const std::vector<std::vector<double>> &params,
        int n, int S, int M, const std::vector<double> &ts, 
        const std::vector<double*> &expM, double tau1, double tau2, int numthreads, double theta, 
        double* outsfs);
void cython_calculate_sfs_jac(const std::vector<std::vector<double>> &params,
        int n, int S, int M, const std::vector<double> &ts, 
        const std::vector<double*> &expM, double tau1, double tau2, int numthreads, double theta, 
        double* outsfs, double* outjac);

void init_eigen();

#endif
