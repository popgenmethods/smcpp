#ifndef MORAN_EIGENSYSTEM_H
#define MORAN_EIGENSYSTEM_H

#include <map>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <gmpxx.h>
#include "mpq_support.h"

typedef struct 
{
    MatrixXq U, Uinv;
    VectorXq D;
} 
MoranEigensystem;

Eigen::SparseMatrix<mpq_class, Eigen::RowMajor> moran_rate_matrix(int);
Eigen::SparseMatrix<mpq_class, Eigen::RowMajor> modified_moran_rate_matrix(int, int);
MoranEigensystem& compute_moran_eigensystem(int);

#endif
