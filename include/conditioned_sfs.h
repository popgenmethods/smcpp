#ifndef CONDITIONED_SFS_H
#define CONDITIONED_SFS_H

#include <exception>
#include <map>

#include "common.h"
#include "piecewise_constant_rate_function.h"
#include "moran_eigensystem.h"
#include "matrix_cache.h"

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
    virtual ~ConditionedSFS() = default;
    virtual std::vector<Matrix<T> > compute(const PiecewiseConstantRateFunction<T> &) const = 0;
};

template <typename T>
class OnePopConditionedSFS : public ConditionedSFS<T>
{
    public:
    OnePopConditionedSFS(int);
    std::vector<Matrix<T> > compute(const PiecewiseConstantRateFunction<T> &) const;

    std::vector<Matrix<T> > compute_below(const PiecewiseConstantRateFunction<T> &) const;
    std::vector<Matrix<T> > compute_above(const PiecewiseConstantRateFunction<T> &) const;

    private:
    const int n;
    const MoranEigensystem mei;
    const MatrixCache mcache;
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
