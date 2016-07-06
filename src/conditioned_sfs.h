#ifndef CONDITIONED_SFS_H
#define CONDITIONED_SFS_H

#include <exception>
#include <map>
#include <random>
#include <gmpxx.h>

#include "common.h"
#include "piecewise_exponential_rate_function.h"
#include "moran_eigensystem.h"
#include "mpq_support.h"
#include "timer.h"

typedef struct 
{
    MatrixXq coeffs;
} below_coeff;

typedef struct { Matrix<double> X0, X2, M0, M1; } MatrixCache;

class improper_sfs_exception : public std::exception
{
    virtual const char* what() const throw()
    {
        return "SFS is not a probability distribution";
    }
};

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
    std::vector<Matrix<T> >& compute(const PiecewiseExponentialRateFunction<T> &);

    // private:
    // Methods
    void construct_ad_vars();
    std::vector<Matrix<T> > compute_below(const PiecewiseExponentialRateFunction<T> &);
    std::vector<Matrix<T> > compute_above(const PiecewiseExponentialRateFunction<T> &);

    // Variables
    const int n, H;
    const MoranEigensystem mei;
    const MatrixCache mcache;
    Matrix<T> tjj_below, M0_below, M1_below;
    std::vector<Matrix<T> > csfs, csfs_below, csfs_above, C_above;
    const Matrix<double> Uinv_mp0, Uinv_mp2;
};

template <typename T>
std::vector<Matrix<T> > incorporate_theta(const std::vector<Matrix<T> > &, const double);

#endif
