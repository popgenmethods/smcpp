#ifndef CONDITIONED_SFS_H
#define CONDITIONED_SFS_H

#include <cassert>
#include <cfenv>
#include <map>
#include <thread>
#include <random>
#include <unsupported/Eigen/MPRealSupport>
#include <mpreal.h>
#include <gmpxx.h>

#include "common.h"
#include "piecewise_exponential_rate_function.h"
#include "matrix_interpolator.h"
#include "ThreadPool.h"

typedef struct 
{
    std::vector<std::valarray<mpq_class>> coeffs;
    mp_prec_t prec;
} below_coeff;

class ConditionedSFSBase
{
    protected:
    static std::map<int, below_coeff> below_coeffs_memo;
};

template <typename T>
class ConditionedSFS : public ConditionedSFSBase
{
    public:
    ConditionedSFS(int, const MatrixInterpolator);
    void compute(const PiecewiseExponentialRateFunction<T>&, int, T, T);
    Matrix<T> matrix() const { return csfs; }
    void set_seed(long long s) { gen.seed(s); }

    // private:
    // Methods
    void fill_matrices();
    void construct_ad_vars();
    Vector<T> compute_etnk_below(const Vector<T>&);
    Vector<T> compute_etnk_below(const std::vector<mpreal_wrapper<T>>&);

    double exp1();
    T exp1_conditional(T, T);
    double unif();
    double rand_exp();

    // Variables
    std::mt19937 gen;
    const int n;
    const MatrixInterpolator moran_interp;
    const below_coeff bc;
    Vector<T> D_subtend_above, D_subtend_below;
    Matrix<T> &Wnbj, &P_dist, &P_undist, csfs, csfs_above, csfs_below, ETnk_below;

    static std::map<int, std::array<Matrix<T>, 3> > matrix_cache;
    static std::array<Matrix<T>, 3>& cached_matrices(int n);
};

template <typename T>
class CSFSManager
{
    public:
    CSFSManager(int n, const MatrixInterpolator &moran_interp, int numthreads, double theta) : theta_(theta), tp_(numthreads)
    {
        for (int i = 0; i < numthreads; ++i)
            csfss.emplace_back(n, moran_interp);
    }

    void set_seed(long long seed)
    {
        gen.seed(seed);
        for (ConditionedSFS<T> &c : csfss)
            c.set_seed(gen());
    }

    Matrix<T> compute(const PiecewiseExponentialRateFunction<T> &eta, int num_samples, double tau1, double tau2)
    {
        std::vector<std::thread> t;
        T t1 = (*eta.getR())(tau1);
        T t2;
        if (std::isinf(tau2))
            t2 = INFINITY;
        else
            t2 = (*eta.getR())(tau2);
        std::vector<std::future<void>> results;
        for (ConditionedSFS<T> &c : csfss)
            results.emplace_back(tp_.enqueue([&c, eta, num_samples, t1, t2] { c.compute(eta, num_samples, t1, t2); }));
        for (auto &res : results) 
            res.wait();
        Eigen::Matrix<T, 3, Eigen::Dynamic> ret = average_csfs();
        if (ret(0,0) <= 0.0 or ret(0.0) >= 1.0)
        {
            std::cout << ret.template cast<double>() << std::endl << std::endl;
            std::cout << t1 << " " << t2 << std::endl << std::endl;
            std::cerr << "sfs is no longer a probability distribution. branch lengths are too long." << std::endl;
        }
        return ret;
        }

    private:
    Matrix<T> average_csfs()
    {
        Matrix<T> ret = Matrix<T>::Zero(csfss[0].matrix().rows(), csfss[0].matrix().cols());
        int m = 0;
        for (const ConditionedSFS<T> &c : csfss)
        {
            ret += c.matrix();
            ++m;
        }
        ret /= (double)m;
        T tauh = ret.sum();
        ret *= -expm1(-theta_ * tauh) / tauh;
        ret(0, 0) = exp(-theta_ * tauh);
        // ret *= theta;
        // ret(0, 0) = 1. - ret.sum();
        return ret;
    }
    std::vector<ConditionedSFS<T>> csfss;
    double theta_;
    ThreadPool tp_;
    std::mt19937 gen;
};

void store_sfs_results(const Matrix<double>&, double*);
void store_sfs_results(const Matrix<adouble>&, double*, double*);

// These methods are used for testing purposes only
void cython_calculate_sfs(const std::vector<std::vector<double>> params,
        int n, int num_samples, const MatrixInterpolator &,
        double tau1, double tau2, int numthreads, double theta, 
        double* outsfs);
void cython_calculate_sfs_jac(const std::vector<std::vector<double>> params,
        int n, int num_samples, const MatrixInterpolator &,
        double tau1, double tau2, int numthreads, double theta, 
        double* outsfs, double* outjac);

void init_eigen();

#endif
