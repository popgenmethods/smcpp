#ifndef MATRIX_EXPONENTIAL_H
#define MATRIX_EXPONENTIAL_H

#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> CMatrix;

void matrix_exponential(const int n, const double* A, double* eA)
{
    auto _A = Eigen::Map<const CMatrix>(A, n, n);
    auto _eA = Eigen::Map<CMatrix>(eA, n, n);
    _eA = _A.exp();
}

#endif
