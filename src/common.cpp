#include "common.h"

std::mutex mtx;
bool do_progress;
void doProgress(bool x) { do_progress = x; }

void store_admatrix(const Matrix<adouble> &M, int nd, double* out, double* outjac)
{
    Matrix<double> M1 = M.cast<double>();
    store_matrix(&M1, out);
    Eigen::VectorXd d;
    unsigned long int m = 0;
    for (int i = 0; i < M.rows(); ++i)
        for (int j = 0; j < M.cols(); ++j)
        {
            d = M(i, j).derivatives();
            for (int k = 0; k < nd; ++k)
                outjac[m++] = d(k);
        }
}

