#ifndef MPREAL_SUPPORT_H
#define MPREAL_SUPPORT_H

#include "mpreal.h"

namespace myfsum
{
    inline mpfr::mpreal fsum(const std::vector<mpfr::mpreal> &v)
    {
        int status;
        return mpfr::sum(v.data(), v.size(), status);
    }
}

// Ugly alias-specialization workaround stuff
template <typename T>
struct mpreal_wrapper_generic {};
template <typename T>
struct mpreal_wrapper_type
{ typedef mpreal_wrapper_generic<T> type; };
template <>
struct mpreal_wrapper_type<double>
{ 
    typedef mpfr::mpreal type; 
    static type convert(const double &x)
    {
        return type(x);
    }
    static double convertBack(const type &x)
    {
        return x.toDouble();
    }
    static type fsum(const std::vector<type> &v)
    {
        return myfsum::fsum(v);
    }
};

template <>
struct mpreal_wrapper_type<adouble>
{ 
    typedef Eigen::Matrix<mpfr::mpreal, Eigen::Dynamic, 1> VectorXmp;
    typedef Eigen::AutoDiffScalar<VectorXmp> type;

    static type convert(const adouble &x)
    {
        return type(x.value(), x.derivatives().template cast<mpfr::mpreal>());
    }
    static adouble convertBack(const type &x)
    {
        return adouble(x.value().toDouble(), x.derivatives().template cast<double>());
    }
    static type fsum(const std::vector<type> &v)
    {
        int nd = v[0].derivatives().rows();
        std::vector<mpfr::mpreal> x; 
        std::vector<std::vector<mpfr::mpreal> > d(nd);
        for (auto &vv : v)
        {
            x.push_back(vv.value());
            for (int i = 0; i < nd; ++i)
                d[i].push_back(vv.derivatives()(i));
        }
        int status;
        type ret = myfsum::fsum(x);
        ret.derivatives() = Vector<mpfr::mpreal>::Zero(nd);
        for (int i = 0; i < nd; ++i)
            ret.derivatives()(i) = myfsum::fsum(d[i]);
        return ret;
    }
};

template <typename T>
using mpreal_wrapper = typename mpreal_wrapper_type<T>::type;
template <typename T>
mpreal_wrapper<T> mpreal_wrapper_convert(const T &x)
{
    return mpreal_wrapper_type<T>::convert(x);
}
template <typename T>
T mpreal_wrapper_convertBack(const mpreal_wrapper<T> &x)
{
    return mpreal_wrapper_type<T>::convertBack(x);
}

inline bool isinf(const mpreal_wrapper<double> &x)
{
    return mpfr::isinf(x);
}

inline bool isinf(const mpreal_wrapper<adouble> &x)
{
    return mpfr::isinf(x.value());
}

namespace Eigen
{
    namespace internal
    {
        template <>
            struct scalar_product_traits<mpfr::mpreal, adouble>
            {
                typedef mpreal_wrapper<adouble> ReturnType;
            };
        template <>
            struct scalar_product_traits<mpfr::mpreal, double>
            {
                typedef mpreal_wrapper<double> ReturnType;
            };
        template <>
            struct cast_impl<mpreal_wrapper<adouble>, adouble>
            {
                static inline adouble run(const mpreal_wrapper<adouble> &x)
                {
                    return adouble(x.value().toDouble(), x.derivatives().template cast<double>());
                }
            };
        template <>
            struct cast_impl<mpreal_wrapper<adouble>, mpfr::mpreal>
            {
                static inline mpfr::mpreal run(const mpreal_wrapper<adouble> &x)
                {
                    return x.value();
                }
            };
    }
}


template <typename T>
inline void print_derivatives(const T&)
{
}

template <>
inline void print_derivatives(const mpreal_wrapper_type<adouble>::type &x)
{
    std::cout << "derivatives: " << x.derivatives().transpose() << std::endl;
}

template <>
inline void print_derivatives(const adouble &x)
{
    std::cout << "derivatives: " << x.derivatives().transpose() << std::endl;
}

inline mpfr::mpreal toMpfr(const mpfr::mpreal &m) { return m; }
inline mpfr::mpreal toMpfr(const mpreal_wrapper<adouble> &m) { return m.value(); }

#endif
