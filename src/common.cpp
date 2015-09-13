#include "common.h"

std::mutex mtx;
bool do_progress;
void doProgress(bool x) { do_progress = x; }

void store_matrix(Matrix<double> *M, double* out)
{
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Map(out, M->rows(), M->cols()) = *M;
}

void store_matrix(Matrix<adouble> *M, double* out)
{
    Matrix<double> MM = M->template cast<double>();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Map(out, MM.rows(), MM.cols()) = MM;
}

void store_admatrix(const Matrix<adouble> &M, int nd, double* out, double* outjac)
{
    Matrix<double> M1 = M.cast<double>();
    store_matrix(&M1, out);
    Eigen::VectorXd d;
    int m = 0;
    for (int i = 0; i < M.rows(); ++i)
        for (int j = 0; j < M.cols(); ++j)
        {
            d = M(i, j).derivatives();
            assert(d.rows() == nd);
            for (int k = 0; k < nd; ++k)
                outjac[m++] = d(k);
        }
}

