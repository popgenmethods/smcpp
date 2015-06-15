#ifndef MATRIX_INTERPOLATOR_H
#define MATRIX_INTERPOLATOR_H

#include "common.h"

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

#endif
