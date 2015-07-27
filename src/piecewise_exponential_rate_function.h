#ifndef PIECEWISE_EXPONENTIAL_RATE_FUNCTION_H
#define PIECEWISE_EXPONENTIAL_RATE_FUNCTION_H

#include <Eigen/Dense>
#include "common.h"
#include "specialfunctions.h"
#include "function_evaluator.h"
#include "sad.h"

template <typename T>
using feval = std::unique_ptr<FunctionEvaluator<T>>;

// Ugly alias-specialization workaround stuff
template <typename T>
struct mpreal_wrapper_generic {};
template <typename T>
struct mpreal_wrapper_type
{ typedef mpreal_wrapper_generic<T> type; };
template <>
struct mpreal_wrapper_type<double>
  { typedef mpfr::mpreal type; };
template <>
struct mpreal_wrapper_type<adouble>
  { typedef sad::simple_autodiff<mpfr::mpreal> type; };
template <typename T>
using mpreal_wrapper = typename mpreal_wrapper_type<T>::type;

template <typename T>
class PiecewiseExponentialRateFunction
{
    public:
    PiecewiseExponentialRateFunction(const std::vector<std::vector<double>>, const std::vector<std::pair<int, int>>);
    PiecewiseExponentialRateFunction(const std::vector<std::vector<double>> params);
    PiecewiseExponentialRateFunction(const PiecewiseExponentialRateFunction &other) : 
        PiecewiseExponentialRateFunction(other.params, other.derivatives) {}
    std::vector<T> getTimes() const { return ts; }
    const FunctionEvaluator<T>* geteta() const { return eta.get(); }
    const FunctionEvaluator<T>* getR() const { return R.get(); }
    const FunctionEvaluator<T>* getRinv() const { return Rinv.get(); }
    void print_debug() const;
    const T regularizer(void) const { return _reg; }
    T tjj_integral(double, T, T, T) const;
    mpreal_wrapper<T> mpfr_tjj_integral(double, T, T, T) const;
    const std::vector<std::pair<int, int>> derivatives;
    const T zero;
    const T one;

    friend std::ostream& operator<<(std::ostream& os, const PiecewiseExponentialRateFunction& pexp)
    {
        os << pexp.ts << std::endl;
        os << pexp.ada << std::endl;
        os << pexp.adb << std::endl;
        return os;
    }
    
    private:
    T init_derivative(double x);
    std::vector<std::vector<double>> params;
    const int K;
    std::vector<T> ada, adb, ads, ts, Rrng;
    void initialize_derivatives();
    void compute_antiderivative();
    feval<T> eta, R, Rinv;
    T _reg;
};

template <typename T>
class BasePExpEvaluator : public FunctionEvaluator<T>
{
    public:
    BasePExpEvaluator(const std::vector<T> ada, const std::vector<T> adb, 
            const std::vector<T> ts, const std::vector<T> Rrng) :
        ada(ada), adb(adb), ts(ts), Rrng(Rrng) {}

    virtual ~BasePExpEvaluator() = default;

    virtual std::vector<T> getTimes(void) const
    {
        return ts;
    }

    virtual T operator()(const T &t) const
    {
        int ip = insertion_point(t, insertion_list(), 0, insertion_list().size());
        return pexp_eval(t, ip);
    }

    virtual std::vector<T> operator()(const std::vector<T> &v) const
    {
        std::vector<T> ret;
        ret.reserve(v.size());
        int ip = insertion_point(v[0], insertion_list(), 0, insertion_list().size());
        for (typename std::vector<T>::const_iterator it = v.begin(); it != v.end(); ++it)
        {
            while (*it > insertion_list()[ip + 1]) ip++;
            ret.push_back(pexp_eval(*it, ip));
        }
        return ret;
    }

    protected:
    const std::vector<T> ada, adb, ts, Rrng;
    virtual T pexp_eval(const T &, int) const = 0;
    virtual const std::vector<T>& insertion_list(void) const = 0;

    friend class PiecewiseExponentialRateFunction<T>;
};

template <typename T>
class PExpEvaluator : public BasePExpEvaluator<T>
{
    public:
    PExpEvaluator(const std::vector<T> ada, const std::vector<T> adb, 
            const std::vector<T> ts, const std::vector<T> Rrng) :
        BasePExpEvaluator<T>(ada, adb, ts, Rrng) {}
    protected:
    virtual const std::vector<T>& insertion_list(void) const { return this->ts; } 
    virtual T pexp_eval(const T &t, int ip) const
    {
        return this->ada[ip] * exp(this->adb[ip] * (t - this->ts[ip]));
    }
};

template <typename T>
class PExpIntegralEvaluator : public BasePExpEvaluator<T>
{
    public:
    PExpIntegralEvaluator(const std::vector<T> ada, const std::vector<T> adb, 
            const std::vector<T> ts, const std::vector<T> Rrng) :
        BasePExpEvaluator<T>(ada, adb, ts, Rrng) {}
    protected:
    virtual const std::vector<T>& insertion_list(void) const { return this->ts; } 
    virtual T pexp_eval(const T &t, int ip) const
    {
        if (this->adb[ip] == 0.0)
            return this->Rrng[ip] + this->ada[ip] * (t - this->ts[ip]);
        else
            return this->ada[ip] / this->adb[ip] * 
                expm1(this->adb[ip] * (t - this->ts[ip])) + this->Rrng[ip];
    }
};

template <typename T>
class PExpInverseIntegralEvaluator : public BasePExpEvaluator<T>
{
    public:
    PExpInverseIntegralEvaluator(const std::vector<T> ada, const std::vector<T> adb, 
            const std::vector<T> ts, const std::vector<T> Rrng) :
        BasePExpEvaluator<T>(ada, adb, ts, Rrng) {}
    private:
    virtual const std::vector<T>& insertion_list(void) const { return this->Rrng; } 
    virtual T pexp_eval(const T &y, int ip) const
    {
        if (this->adb[ip] == 0.0) 
            return (y - this->Rrng[ip]) / this->ada[ip] + this->ts[ip];
        else
            return log1p((y - this->Rrng[ip]) * this->adb[ip] / this->ada[ip]) / this->adb[ip] + this->ts[ip];
    }
};

#endif
