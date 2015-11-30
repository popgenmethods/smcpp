#ifndef CONDITIONED_SFS_H
#define CONDITIONED_SFS_H

#include <cassert>
#include <cfenv>
#include <map>
#include <thread>
#include <random>
#include <gmpxx.h>

#include "common.h"
#include "piecewise_exponential_rate_function.h"
#include "moran_eigensystem.h"
#include "mpq_support.h"

typedef struct 
{
    MatrixXq coeffs;
} below_coeff;

typedef struct { Matrix<double> X0, X2, M0, M1; } MatrixCache;

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
    void compute_below(const PiecewiseExponentialRateFunction<T> &);
    void compute_above(const PiecewiseExponentialRateFunction<T> &);

    // Variables
    const int n, H;
    const MoranEigensystem mei;
    const MatrixCache mcache;
    Matrix<T> tjj_below, M0_below, M1_below;
    std::vector<Matrix<T> > csfs, csfs_below, csfs_above, C_above;
};

#endif
