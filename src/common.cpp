#include <execinfo.h>
#include <signal.h>
#include <unistd.h>

#include "common.h"

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
    if (Logger::logger_cb)
        Logger::logger_cb(name, level, message);
}

Logger::Logger(const char* name, int line, const char* level) : name(name), line(line), level(level) {}
Logger::~Logger()
{
    oss << name << ":" << line;
    call_logger(oss.str().c_str(), level, stream.str().c_str());
}

void signal_bt(int sig)
{
#pragma omp critical(stacktrace)
    print_stacktrace();
}

void init_logger_cb(void(*fp)(const char*, const char*, const char*))
{
    Logger::logger_cb = fp;
    signal(SIGSEGV, signal_bt);
    signal(SIGABRT, signal_bt);
}

ParameterVector shiftParams(const ParameterVector &model1, const double shift)
{
    const std::vector<adouble> &a = model1[0];
    const std::vector<adouble> &s = model1[1];
    std::vector<adouble> cs(s.size() + 1);
    cs[0] = 0.;
    std::partial_sum(s.begin(), s.end(), cs.begin() + 1);
    cs.back() = INFINITY;
    adouble tshift = shift;
    int ip = std::distance(cs.begin(), std::upper_bound(cs.begin(), cs.end(), tshift)) - 1;
    std::vector<adouble> sp(s.begin() + ip, s.end());
    sp[0] = cs[ip + 1] - shift;
    sp.back() = 1.0;
    std::vector<adouble> ap(a.begin() + ip, a.end());
    return {ap, sp};
}

ParameterVector truncateParams(const ParameterVector params, const double truncationTime)
{
    const std::vector<adouble> &a = params[0];
    const std::vector<adouble> &s = params[1];
    std::vector<adouble> cs(s.size() + 1);
    cs[0] = 0.;
    std::partial_sum(s.begin(), s.end(), cs.begin() + 1);
    cs.back() = INFINITY;
    adouble tt = truncationTime;
    int ip = std::distance(cs.begin(), std::upper_bound(cs.begin(), cs.end(), tt)) - 1;
    std::vector<adouble> sp(s.begin(), s.begin() + ip + 1);
    sp[ip] = truncationTime - cs[ip];
    std::vector<adouble> ap(a.begin(), a.begin() + ip + 1);
    sp.push_back(1.);
    ap.push_back(1e-8);
    return {ap, sp};
}

void check_nan(const double x, const char* file, const int line) 
{ 
    std::string s;
    if (std::isnan(x))
        s = "nan";
    else if (std::isinf(x))
        s = "inf";
    else
        return;
    s += " detected at ";
    s += file;
    s += ":";
    s += std::to_string(line);
#pragma omp critical(stacktrace)
    {
        CRITICAL << s;
        print_stacktrace();
    }
    throw std::runtime_error(s);
}

void check_nan(const adouble &x, const char* file, const int line)
{
    check_nan(x.value(), file, line);
    check_nan(x.derivatives(), file, line);
}
