#ifndef CONDITIONED_SFS_H
#define CONDITIONED_SFS_H

#include <exception>
#include <map>
#include <random>
#include <gmpxx.h>

#include "common.h"
#include "piecewise_constant_rate_function.h"
#include "moran_eigensystem.h"
#include "mpq_support.h"

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

template <typename T>
class ConditionedSFS
{
    public:
    virtual std::vector<Matrix<T> > compute(const PiecewiseConstantRateFunction<T> &) const = 0;
};

template <typename T>
class OnePopConditionedSFS : public ConditionedSFS<T>
{
    public:
    OnePopConditionedSFS(int, int);
    std::vector<Matrix<T> > compute(const PiecewiseConstantRateFunction<T> &) const;

    // private:
    std::vector<Matrix<T> > compute_below(const PiecewiseConstantRateFunction<T> &) const;
    std::vector<Matrix<T> > compute_above(const PiecewiseConstantRateFunction<T> &) const;

    // Variables
    static MatrixCache& cached_matrices(int n);
    static std::map<int, MatrixCache> matrix_cache;

    const int n, H;
    const MoranEigensystem mei;
    const MatrixCache mcache;
    // std::vector<Matrix<T> > csfs, csfs_below, csfs_above, C_above;
    const Matrix<double> Uinv_mp0, Uinv_mp2;
};

template <typename T>
class DummySFS : public ConditionedSFS<T>
{
    public:
    DummySFS(const int dim, const int M, const std::vector<double*> sfs) : dim(dim), M(M), precomputedSFS(storeSFS(sfs)) {}

    std::vector<Matrix<T> > compute(const PiecewiseConstantRateFunction<T> &) const
    {
        return precomputedSFS;
    }

    private:
    std::vector<Matrix<T> > storeSFS(const std::vector<double*> newSFS)
    {
        std::vector<Matrix<T> > ret(M, Matrix<T>::Zero(3, dim));
        for (int m = 0; m < M; ++m)
            ret[m] = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Map(newSFS[m], 3, dim).template cast<adouble>();
        return ret;
    }

    const int dim, M;
    const std::vector<Matrix<T> > precomputedSFS;
};



template <typename T>
std::vector<Matrix<T> > incorporate_theta(const std::vector<Matrix<T> > &, const double);

#endif
