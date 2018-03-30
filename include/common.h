#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <ostream>
#include <vector>
#include <array>

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
  inline const Eigen::AutoDiffScalar< \
  EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE(typename Eigen::internal::remove_all<DerType>::type, typename Eigen::internal::traits<typename Eigen::internal::remove_all<DerType>::type>::Scalar, product) > \
  FUNC(const Eigen::AutoDiffScalar<DerType>& x) { \
    using namespace Eigen; \
    EIGEN_UNUSED typedef typename Eigen::internal::traits<typename Eigen::internal::remove_all<DerType>::type>::Scalar Scalar; \
    CODE; \
  }

using std::log1p;
using std::cosh;
using std::sinh;
using mpfr::exp;
using mpfr::sinh;
using mpfr::cosh;

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(expm1,
  using std::exp;
  using std::expm1;
  Scalar expm1x = expm1(x.value());
  Scalar expx = exp(x.value());
  return Eigen::MakeAutoDiffScalar(expm1x, x.derivatives() * expx);
)

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(log1p,
  using std::log1p;
  Scalar log1px = log1p(x.value());
  return Eigen::MakeAutoDiffScalar(log1px, x.derivatives() * (Scalar(1) / (Scalar(1) + x.value())));
)

#undef EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY

};

inline void init_eigen() { Eigen::initParallel(); }

void store_matrix(const Matrix<double> &M, double* out);
void store_matrix(const Matrix<adouble> &M, double* out);
void store_matrix(const Matrix<adouble> &M, double *out, double *jac);

void init_logger_cb(void(*)(const std::string, const std::string, const std::string));
void call_logger(const std::string, const std::string, const std::string);
struct Logger
{
    static void(*logger_cb)(const std::string, const std::string, const std::string);
    Logger(const std::string name, int line, const std::string level);
    const std::string name;
    const int line;
    const std::string level;
    const std::ostringstream prefix;
    std::ostringstream stream;
    template <typename T>
    Logger& operator<<(const T &data)
    {
        (stream) << data;
        return *this;
    }
    virtual ~Logger();

    private:
    Logger(const Logger&);
    Logger& operator=(const Logger&);
};

#define DEBUG Logger(__FILE__, __LINE__, "DEBUG")
#define DEBUG1 Logger(__FILE__, __LINE__, "DEBUG1")
#define INFO Logger(__FILE__, __LINE__, "INFO")
#define WARNING Logger(__FILE__, __LINE__, "WARNING")
#define CRITICAL Logger(__FILE__, __LINE__, "CRITICAL")
#define ERROR Logger(__FILE__, __LINE__, "ERROR")

#ifdef NO_CHECK_NAN
#define CHECK_NAN(x)
#define CHECK_NAN_OR_NEGATIVE(x)
#else
#define CHECK_NAN(x) check_nan(x, __FILE__, __LINE__)
#define CHECK_NAN_OR_NEGATIVE(x) check_nan(x, __FILE__, __LINE__); check_negative(x, __FILE__, __LINE__);
#endif

inline void fill_jacobian(const adouble &ll, double* outjac)
{
    adouble_t d = ll.derivatives();
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>, Eigen::RowMajor> _jac(outjac, d.rows());
    _jac = d.template cast<double>();
}

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

void check_negative(const adouble x, const char* file, const int line);
void check_negative(const double x, const char* file, const int line);

template <typename Derived>
void check_negative(const Eigen::DenseBase<Derived> &M, const char* file, const int line)
{
    for (int i = 0; i < M.rows(); ++i)
        for (int j = 0; j < M.cols(); ++j)
        {
            try 
            {
                check_negative(M.coeff(i, j), file, line);
            }
            catch (std::runtime_error)
            {
                CRITICAL << "rows:" << M.rows() << " cols:" << M.cols() 
                         << " M(" << i << "," << j << ")=" << M.coeff(i, j);
                throw;
            }
        }
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
