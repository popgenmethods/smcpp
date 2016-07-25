#ifndef MORAN_EIGENSYSTEM_H
#define MORAN_EIGENSYSTEM_H

#include <Eigen/Sparse>

#include "mpq_support.h"

struct MoranEigensystem
{
    MoranEigensystem(const int n) : U(n + 1, n + 1), Uinv(n + 1, n + 1), D(n + 1) 
    {
        U.setZero();
        Uinv.setZero();
        D.setZero();
    }
    MatrixXq U, Uinv;
    VectorXq D;
};

Eigen::SparseMatrix<mpq_class, Eigen::RowMajor> moran_rate_matrix(int);
Eigen::SparseMatrix<mpq_class, Eigen::RowMajor> modified_moran_rate_matrix(int, int);
MoranEigensystem& compute_moran_eigensystem(int);

#endif
