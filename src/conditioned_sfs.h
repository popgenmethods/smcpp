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

class MatrixInterpolator
{
    public:
    MatrixInterpolator(int n, const std::vector<double> &ts, const std::vector<double*> expM) : 
        n(n), ts(ts), expM(expM)
    {
        for (auto ep : expM)
        {
            const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> em = 
                Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Map(ep, n, n);
            expM_map.push_back(em);
        }
    }
    MatrixInterpolator(const MatrixInterpolator &other) : 
        MatrixInterpolator(other.n, other.ts, other.expM) {}

    template <typename T>
    Matrix<T> interpolate(T t) const
    {
        int ei = insertion_point(toDouble(t), ts, 0, ts.size());
        T coef = (t - ts[ei]) / (ts[ei + 1] - ts[ei]);
        Matrix<T> ret = (1 - coef) * expM_map[ei].template cast<T>() + coef * expM_map[ei + 1].template cast<T>();
        return ret;
    }

    private:
    const int n;
    const std::vector<double> ts;
    const std::vector<double*> expM;
    // const std::vector<Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>> expM_map;
    std::vector<Matrix<double>> expM_map;
};

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
