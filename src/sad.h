#ifndef SAD_H
#define SAD_H

#include "mpreal.h"


namespace sad
{
    template <typename T>
    class simple_autodiff;

    template <typename T>
    simple_autodiff<T> exp(const simple_autodiff<T> &s)
    {
        return simple_autodiff<T>(exp(s.x), exp(s.x) * s.d);
    }
    template <typename T>
    simple_autodiff<T> expm1(const simple_autodiff<T> &s)
    {
        return simple_autodiff<T>(expm1(s.x), exp(s.x) * s.d);
    }
    template <typename T>
    simple_autodiff<T> log(const simple_autodiff<T> &s)
    {
        return simple_autodiff<T>(log(s.x), s.d / s.x);
    }
    template <typename T>
    simple_autodiff<T> eint(const simple_autodiff<T> &s)
    {
        return simple_autodiff<T>(eint(s.x), exp(s.x) / s.x * s.d);
    }
    template <typename T>
    bool isinf(const simple_autodiff<T> &s)
    {
        return isinf(s.x);
    }

    class simple_autodiff_base
    {
        public:
        static int nd;
    };

    template <typename T>
    class simple_autodiff : simple_autodiff_base
    {
        public:
        simple_autodiff() : simple_autodiff(0.0) {}
        simple_autodiff(adouble x) : simple_autodiff(T(x.value()), x.derivatives()) {}
        simple_autodiff(double x) : simple_autodiff(T(x), std::valarray<T>(T(0.0), nd)) {}
        simple_autodiff(T x) : simple_autodiff(x, std::valarray<T>(T(0.0), nd)) {}
        simple_autodiff(T x, std::valarray<T> d) : x(x), d(d) {}
        simple_autodiff(T x, Eigen::VectorXd deriv) : simple_autodiff<T>(x, vec2va(deriv)) {}

        friend std::ostream& operator<< (std::ostream &out, simple_autodiff &x)
        {
            out << x.x << "::[";
            for (int i = 0; i < nd - 1; ++i)
                out << x.d[i] << ",";
            out << x.d[nd - 1] << "]";
            return out;
        }

        adouble toDouble()
        {
            Vector<double> deriv(nd);
            for (int i = 0; i < nd; ++i)
                deriv(i) = d[i].toDouble();
            return adouble(x.toDouble(), deriv);
        }

        simple_autodiff<T>& operator+=(const simple_autodiff<T>& rhs)
        {
            x += rhs.x;
            d += rhs.d;
            return *this;
        }
        simple_autodiff<T>& operator-=(const simple_autodiff<T>& rhs)
        {
            x -= rhs.x;
            d -= rhs.d;
            return *this;
        }
        friend simple_autodiff<T> operator-(simple_autodiff<T> lhs)
        {
            return simple_autodiff<T>(-lhs.x, -lhs.d);
        }
        friend simple_autodiff<T> operator-(simple_autodiff<T> lhs, const simple_autodiff<T> &rhs)
        {
            return simple_autodiff<T>(lhs.x - rhs.x, lhs.d - rhs.d);
        }
        friend simple_autodiff<T> operator+(simple_autodiff<T> lhs, const simple_autodiff<T> &rhs)
        {
            return simple_autodiff<T>(lhs.x + rhs.x, lhs.d + rhs.d);
        }
        friend simple_autodiff<T> operator*(simple_autodiff<T> lhs, const simple_autodiff<T> &rhs)
        {
            return simple_autodiff<T>(lhs.x * rhs.x, lhs.x * rhs.d + lhs.d * rhs.x);
        }
        friend simple_autodiff<T> operator/(simple_autodiff<T> lhs, const simple_autodiff<T> &rhs)
        {
            return simple_autodiff<T>(lhs.x / rhs.x, (rhs.x * lhs.d - lhs.x * rhs.d) / (rhs.x * rhs.x));
        }
        friend bool operator<(const simple_autodiff<T> &lhs, const simple_autodiff<T> &rhs)
        {
            return lhs.x < rhs.x;
        }
        friend bool operator>(const simple_autodiff<T> &lhs, const simple_autodiff<T> &rhs)
        {
            return lhs.x > rhs.x;
        }
        friend bool operator==(const simple_autodiff<T> &lhs, const simple_autodiff<T> &rhs)
        {
            return lhs.x == rhs.x;
        }

        friend simple_autodiff<T> sad::exp<T>(const simple_autodiff<T> &s);
        friend simple_autodiff<T> sad::expm1<T>(const simple_autodiff<T> &s);
        friend simple_autodiff<T> sad::eint<T>(const simple_autodiff<T> &s);
        friend bool sad::isinf<T>(const simple_autodiff<T> &s);

        private:
        std::valarray<T> vec2va(Eigen::VectorXd deriv)
        {
            std::valarray<T> ret(deriv.size());
            for (int i = 0; i < deriv.size(); ++i)
                ret[i] = deriv[i];
            return ret;
        }
        T x;
        std::valarray<T> d;
    };
}
#endif
