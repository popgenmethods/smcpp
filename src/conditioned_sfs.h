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
#include "moran_eigensystem.h"
#include "mpq_support.h"

typedef struct 
{
    MatrixXq coeffs;
    mp_prec_t prec;
} below_coeff;

typedef struct { MatrixXq X0, X2, M0, M1; mp_prec_t prec; } MatrixCache;

class ConditionedSFSBase
{
    protected:
    static std::map<int, below_coeff> below_coeffs_memo;
    static MatrixCache& cached_matrices(int n);
    static std::map<int, MatrixCache> matrix_cache;
};

template <typename T>
class ConditionedSFS : public ConditionedSFSBase
{
    public:
    ConditionedSFS(int, int);
    std::vector<Matrix<T> >& compute(const PiecewiseExponentialRateFunction<T> &, double);

    // private:
    // Methods
    void construct_ad_vars();
    Matrix<T> above0(const Matrix<T>&);
    Matrix<T> above2(const Matrix<T>&);
    Matrix<T> below0(const Matrix<mpreal_wrapper<T> >&);
    Matrix<T> below1(const Matrix<mpreal_wrapper<T> >&);
    template <typename Derived>
    Matrix<T> parallel_cwiseProduct_colSum(const MatrixXq &a, const Eigen::MatrixBase<Derived> &b);
    template <typename Derived>
    void parallel_matrix_product(const Eigen::MatrixBase<Derived> &, const MatrixXq &, Matrix<T> &ret);
    // Vector<T> compute_etnk_below(const Vector<T>&);
    // Vector<T> compute_etnk_below(const std::vector<mpreal_wrapper<T> >&);
    Matrix<T> compute_etnk_below_mat(const Matrix<mpreal_wrapper<T> >&);
    void compute_below(const PiecewiseExponentialRateFunction<T> &);
    void compute_above(const PiecewiseExponentialRateFunction<T> &);

    // Variables
    const int n, H;
    const MoranEigensystem mei;
    const MatrixCache mcache;
    std::vector<std::vector<std::vector<mpreal_wrapper<T> > > > vs;
    Matrix<mpreal_wrapper<T> > tjj_below;
    Matrix<T> M0_below, M1_below;
    std::vector<Matrix<T> > csfs_below, csfs_above, csfs, C_above;
};

#endif
