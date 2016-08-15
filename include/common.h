#ifndef COMMON_H
#define COMMON_H

#include <mutex>
#include <iostream>
#include <vector>
#include <random>
#include <array>
#include <ostream>

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/AutoDiff>
#include "mpreal.h"

#include "prettyprint.h"
#include "hash.h"
#include "stacktrace.h"

template <typename T> using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template <typename T> using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template <typename T, size_t P> using FixedVector = Eigen::Matrix<T, P, 1>;

typedef double adouble_base_type;
typedef Eigen::Matrix<adouble_base_type, Eigen::Dynamic, 1> adouble_t;
typedef Eigen::AutoDiffScalar<adouble_t> adouble;
typedef std::vector<std::vector<adouble>> ParameterVector;

template <typename T>
inline T doubly_compensated_summation(const std::vector<T> &x)
{
    if (x.size() == 0)
        return 0.0;
    T s = x[0];
    T c = 0.0;
    T y, u, v, t, z;
    for (unsigned int i = 1; i < x.size(); ++i)
    {
        y = c + x[i];
        u = x[i] - (y - c);
        t = y + s;
        v = y - (t - s);
        z = u + v;
        s = t + z;
        c = z - (s - t);
    }
    return s;
}

inline double toDouble(const adouble &a) { return a.value(); }
inline double toDouble(const double &d) { return d; }

namespace Eigen {
    // Allow for casting of adouble matrices to double
    namespace internal 
    {
        template <>
            struct cast_impl<adouble, float>
            {
                static inline double run(const adouble &x)
                {
                    return static_cast<float>(x.value());
                }
            };
        template <>
            struct cast_impl<adouble, double>
            {
                static inline double run(const adouble &x)
                {
                    return x.value();
                }
            };
    }

// Copied from Eigen's AutoDiffScalar.h
#define EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(FUNC,CODE) \
  template<typename DerType> \
  inline const Eigen::AutoDiffScalar<Eigen::CwiseUnaryOp<Eigen::internal::scalar_multiple_op<typename Eigen::internal::traits<typename Eigen::internal::remove_all<DerType>::type>::Scalar>, const typename Eigen::internal::remove_all<DerType>::type> > \
  FUNC(const Eigen::AutoDiffScalar<DerType>& x) { \
    using namespace Eigen; \
    typedef typename Eigen::internal::traits<typename Eigen::internal::remove_all<DerType>::type>::Scalar Scalar; \
    typedef AutoDiffScalar<CwiseUnaryOp<Eigen::internal::scalar_multiple_op<Scalar>, const typename Eigen::internal::remove_all<DerType>::type> > ReturnType; \
    CODE; \
  }

using std::exp;
using std::expm1;
using std::log1p;
using std::cosh;
using std::sinh;
using mpfr::exp;
using mpfr::sinh;
using mpfr::cosh;

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(expm1,
  Scalar expm1x = expm1(x.value());
  Scalar expx = exp(x.value());
  return ReturnType(expm1x, x.derivatives() * expx);
)

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(ceil,
  Scalar ceil = std::ceil(x.value());
  return ReturnType(ceil, x.derivatives() * Scalar(0));
)

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(log1p,
  Scalar log1px = std::log1p(x.value());
  return ReturnType(log1px, x.derivatives() * (Scalar(1) / (Scalar(1) + x.value())));
)

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(sinh,
  Scalar sinhx = sinh(x.value());
  Scalar coshx = cosh(x.value());
  return ReturnType(sinhx, x.derivatives() * coshx);
)

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(cosh,
  Scalar sinhx = sinh(x.value());
  Scalar coshx = cosh(x.value());
  return ReturnType(coshx, x.derivatives() * sinhx);
)

#undef EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY

};

inline void init_eigen() { Eigen::initParallel(); }

inline void fill_jacobian(const adouble &ll, double* outjac)
{
    adouble_t d = ll.derivatives();
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>, Eigen::RowMajor> _jac(outjac, d.rows());
    _jac = d.template cast<double>();
}

void store_matrix(const Matrix<double> &M, double* out);
void store_matrix(const Matrix<adouble> &M, double* out);
void store_matrix(const Matrix<adouble> &M, double *out, double *jac);

void init_logger_cb(void(*)(const char*, const char*, const char*));
void call_logger(const char*, const char*, const char*);
struct Logger : public std::ostream
{
    static void(*logger_cb)(const char*, const char*, const char*);
    Logger(const char* name, int line, const char* level) : name(name), line(line), level(level) {}
    const char* name;
    const int line;
    const char* level;
    std::stringstream stream;
    template <typename T>
    Logger &operator<<(const T &data)
    {
        (stream) << data;
        return *this;
    }
    ~Logger()
    {
        std::ostringstream oss;
        oss << name << ":" << line;
        call_logger(oss.str().c_str(), level, stream.str().c_str());
    }
};

#define DEBUG Logger(__FILE__, __LINE__, "DEBUG")
#define DEBUG1 Logger(__FILE__, __LINE__, "DEBUG1")
#define INFO Logger(__FILE__, __LINE__, "INFO")
#define WARNING Logger(__FILE__, __LINE__, "WARNING")
#define CRITICAL Logger(__FILE__, __LINE__, "CRITICAL")
#define ERROR Logger(__FILE__, __LINE__, "ERROR")
#define CHECK_NAN(x) check_nan(x, __FILE__, __LINE__)
#define CHECK_NAN_OR_NEGATIVE(x) check_nan(x, __FILE__, __LINE__); check_negative(x, __FILE__, __LINE__);

void check_nan(const double x, const char* file, const int line);
void check_nan(const adouble &x, const char* file, const int line);

template <typename Derived>
void check_nan(const Eigen::DenseBase<Derived> &M, const char* file, const int line)
{
    for (int i = 0; i < M.rows(); ++i)
        for (int j = 0; j < M.cols(); ++j)
        {
            try 
            {
                check_nan(M.coeff(i, j), file, line);
            }
            catch (std::runtime_error)
            {
                CRITICAL << "rows:" << M.rows() << " cols:" << M.cols() 
                         << " M(" << i << "," << j << ")=" << M.coeff(i, j);
                throw;
            }
        }
}

template <typename T>
void check_negative(const T x, const char* file, const int line)
{
    if (x > -1e-16)
        return;
    std::string s = "negative value detected at ";
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

inline adouble double_vec_to_adouble(const double &x, const std::vector<double> &dx)
{
    adouble ret;
    adouble_t vdx(dx.size());
    for (unsigned int i = 0; i < dx.size(); ++i)
        vdx(i) = dx[i];
    return adouble(x, vdx);
}

ParameterVector truncateParams(const ParameterVector params, const double truncationTime);
ParameterVector shiftParams(const ParameterVector &model1, const double shift);

#endif
