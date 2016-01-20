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

void (*Logger::logger_cb)(const char*, const char*, const char*) = 0;
void call_logger(const char* name, const char* level, const char* message)
{
    Logger::logger_cb(name, level, message);
}

void init_logger_cb(void(*fp)(const char*, const char*, const char*))
{
    Logger::logger_cb = fp;
}
