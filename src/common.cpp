#include "common.h"

std::mutex mtx;
bool do_progress;
void doProgress(bool x) { do_progress = x; }

void store_matrix(const Matrix<double> &M, double* out)
{
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Map(out, M.rows(), M.cols()) = M;
}

void store_matrix(const Matrix<adouble> &M, double* out)
{
    Matrix<double> Md = M.cast<double>();
    store_matrix(Md, out);
}

void store_matrix(const Matrix<adouble> &M, double *out, double *outjac)
{
    Matrix<double> Md = M.cast<double>();
    store_matrix(Md, out);
    Eigen::VectorXd d;
    unsigned long int m = 0;
    for (int i = 0; i < M.rows(); ++i)
        for (int j = 0; j < M.cols(); ++j)
        {
            d = M(i, j).derivatives();
            int nd = d.size();
            for (int k = 0; k < nd; ++k)
                outjac[m++] = d(k);
        }
}

void (*Logger::logger_cb)(const char*, const char*, const char*) = 0;
void call_logger(const char* name, const char* level, const char* message)
{
#pragma omp critical(logger_cb)
    {
        Logger::logger_cb(name, level, message);
    }
}

void init_logger_cb(void(*fp)(const char*, const char*, const char*))
{
    Logger::logger_cb = fp;
}
