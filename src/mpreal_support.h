#ifndef MPREAL_SUPPORT_H
#define MPREAL_SUPPORT_H

#include "mpreal.h"

namespace myfsum
{
    inline mpfr::mpreal fsum(const std::vector<mpfr::mpreal> &v, const int n)
    {
        mpfr_srcptr p[n];
        int prec = (int)v[0].getPrecision();
        for (unsigned long int i = 0; i < n; i++) 
        {
            p[i] = v[i].mpfr_srcptr();
            prec = std::min(prec, (int)v[i].getPrecision());
        }
        mpfr::mpreal ret(0.0, prec);
        mpfr_sum(ret.mpfr_ptr(), (mpfr_ptr*)p, n, MPFR_RNDN);
        return ret;
    }
    inline mpfr::mpreal fsum(const std::vector<mpfr::mpreal> &v)
    {
        return fsum(v, v.size());
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
    static type fsum(const std::vector<type> &v, const int n)
    {
        return myfsum::fsum(v, n);
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
    static type fsum(const std::vector<type> &v, const int n)
    {
        int nd = v[0].derivatives().rows();
        std::vector<mpfr::mpreal> x; 
        std::vector<std::vector<mpfr::mpreal> > d(nd);
        for (int i = 0; i < n; ++i)
        {
            x.push_back(v[i].value());
            for (int j = 0; j < nd; ++j)
                d[j].push_back(v[i].derivatives()(j));
        }
        int status;
        type ret = myfsum::fsum(x);
        ret.derivatives() = Vector<mpfr::mpreal>::Zero(nd);
        for (int i = 0; i < nd; ++i)
            ret.derivatives()(i) = myfsum::fsum(d[i]);
        return ret;
    }
    static type fsum(const std::vector<type> &v)
    {
        return fsum(v, v.size());
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
            struct cast_impl<adouble, mpreal_wrapper<adouble> >
            {
                static inline mpreal_wrapper<adouble> run(const adouble &x)
                {
                    return mpreal_wrapper<adouble>(x.value(), x.derivatives().template cast<mpfr::mpreal>());
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
        template <>
            struct cast_impl<double, mpfr::mpreal>
            {
                static inline mpfr::mpreal run(const double &x)
                {
                    return x;
                }
            };
    }
}

inline mpreal_wrapper<adouble> myabs(const mpreal_wrapper<adouble> &a)
{
    return Eigen::abs(a);
}

#endif
