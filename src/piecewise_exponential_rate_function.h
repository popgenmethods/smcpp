#ifndef PIECEWISE_EXPONENTIAL_RATE_FUNCTION_H
#define PIECEWISE_EXPONENTIAL_RATE_FUNCTION_H

#include <random>
#include "common.h"
#include "function_evaluator.h"
#include "mpq_support.h"
#include "exponential_integrals.h"

template <typename T>
class ConditionedSFS;

template <typename T>
class HJTransition;

template <typename T>
using feval = std::unique_ptr<FunctionEvaluator<T>>;

template <typename T>
class PiecewiseExponentialRateFunction
{
    public:
    PiecewiseExponentialRateFunction(const std::vector<std::vector<adouble>>, const std::vector<double>,
            const std::vector<double>);
    PiecewiseExponentialRateFunction(const PiecewiseExponentialRateFunction &other) : 
        PiecewiseExponentialRateFunction(other.params, other.s, other.hidden_states) {}
    const FunctionEvaluator<T>* geteta() const { return _eta.get(); }
    const FunctionEvaluator<T>* getR() const { return _R.get(); }
    const FunctionEvaluator<T>* getRinv() const { return _Rinv.get(); }
    T R(T x) const { T ret = (*_R)(x); check_nan(ret); return ret; }
    T eta(T x) const { return (*_eta)(x); }
    T R_integral(const T, const T) const;
    T R_integral(const T, const T, const T) const;
    std::vector<T> average_coal_times() const;
    void print_debug() const;
    
    Matrix<T> tjj_all_above(const int, const MatrixXq&, const MatrixXq&, const MatrixXq&, const MatrixXq&) const;
    void tjj_double_integral_above(const int, long, std::vector<Matrix<T> > &) const;
    void tjj_double_integral_below(const int, const int, Matrix<T>&) const;
    double random_time(const double, const double, const int) const;
    T random_time(const double, const T&, const T&, std::mt19937&) const;

    friend class ConditionedSFS<T>;
    friend class HJTransition<T>;

    friend std::ostream& operator<<(std::ostream& os, const PiecewiseExponentialRateFunction& pexp)
    {
        os << pexp.ts << std::endl;
        os << pexp.ada << std::endl;
        os << pexp.adb << std::endl;
        return os;
    }
    
    private:
    std::vector<std::vector<adouble>> params;
    const int nder;
    
    private:
    int K;
    std::vector<T> ada, adb, ts, Rrng;
    std::vector<double> s;
    void compute_antiderivative();
    feval<T> _eta, _R, _Rinv;
    std::vector<int> hs_indices;

    public:
    const std::vector<double> hidden_states;
    const double tmax;
    const T zero, one;
};

template <typename T>
class BasePExpEvaluator : public FunctionEvaluator<T>
{
    public:
    BasePExpEvaluator(const std::vector<T> ada, const std::vector<T> adb, 
            const std::vector<T> ts, const std::vector<T> Rrng) :
        ada(ada), adb(adb), ts(ts), Rrng(Rrng) {}

    virtual ~BasePExpEvaluator() = default;

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
        T ret;
        if (this->adb[ip] == 0.0)
            ret = this->Rrng[ip] + this->ada[ip] * (t - this->ts[ip]);
        else
            ret = this->ada[ip] / this->adb[ip] * 
                expm1(this->adb[ip] * (t - this->ts[ip])) + this->Rrng[ip];
        check_nan(ret);
        return ret;
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
