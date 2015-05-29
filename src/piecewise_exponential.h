#ifndef PIECEWISE_EXPONENTIAL_H
#define PIECEWISE_EXPONENTIAL_H

#include <Eigen/Dense>
#include "common.h"
#include "rate_function.h"

template <typename T>
class PiecewiseExponentialRateFunction : public RateFunction<T>
{
    public:
    PiecewiseExponentialRateFunction(const std::vector<std::vector<double>> &params);
    virtual const FunctionEvaluator<T>* getR() const { return R.get(); }
    virtual const FunctionEvaluator<T>* getRinv() const { return Rinv.get(); }
    void print_debug() const;

    friend std::ostream& operator<<(std::ostream& os, const PiecewiseExponentialRateFunction& pexp)
    {
        os << pexp.ts << std::endl;
        os << pexp.ada << std::endl;
        os << pexp.adb << std::endl;
        return os;
    }
    
    private:
    std::vector<T> ada, adb, ads, ts, Rrng;
    const int K;
    void initialize_derivatives();
    void compute_antiderivative();
    feval<T> R, Rinv;
};

template <typename T>
class BasePExpEvaluator : public FunctionEvaluator<T>
{
    public:
    BasePExpEvaluator(const std::vector<T> &ada, const std::vector<T> &adb, 
            const std::vector<T> &ts, const std::vector<T> &Rrng) :
        ada(ada), adb(adb), ts(ts), Rrng(Rrng), K(ts.size()) {}

    virtual T operator()(const T &t) const
    {
        int ip = insertion_point(t, ts, 0, K);
        return pexp_eval(t, ip);
    }

    virtual std::vector<T> operator()(const std::vector<T> &v) const
    {
        std::vector<T> ret(v.size());
        int ip = insertion_point(v[0], ts, 0, K);
        for (typename std::vector<T>::const_iterator it = std::next(v.begin()); it != v.end(); ++it)
        {
            if (*(it - 1) > *it)
                throw std::domain_error("vector must be sorted");
            while (*it > ts[ip + 1]) ip++;
            ret.push_back(pexp_eval(*it, ip));
        }
        return ret;
    }

    protected:
    const std::vector<T> ada, adb, ts, Rrng;
    const int K;
    virtual T pexp_eval(const T &, int) const = 0;
};

template <typename T>
class PExpIntegralEvaluator : public BasePExpEvaluator<T>
{
    public:
    PExpIntegralEvaluator(const std::vector<T> &ada, const std::vector<T> &adb, 
            const std::vector<T> &ts, const std::vector<T> &Rrng) :
        BasePExpEvaluator<T>(ada, adb, ts, Rrng) {}
    private:
    virtual T pexp_eval(const T &t, int ip) const
    {
        if (this->adb[ip] == 0.0)
            return this->Rrng[ip] + this->ada[ip] * (t - this->ts[ip]);
        else
            return this->ada[ip] / this->adb[ip] * expm1(this->adb[ip] * (t - this->ts[ip])) + this->Rrng[ip];
    }
};

template <typename T>
class PExpInverseIntegralEvaluator : BasePExpEvaluator<T>
{
    public:
    PExpInverseIntegralEvaluator(const std::vector<T> &ada, const std::vector<T> &adb, 
            const std::vector<T> &ts, const std::vector<T> &Rrng) :
        BasePExpEvaluator<T>(ada, adb, ts, Rrng) {}
    private:
    virtual T pexp_eval(const T &y, int ip) const
    {
        if (this->adb[ip] == 0.0) 
            return (y - this->Rrng[ip]) / this->ada[ip] + this->ts[ip];
        else
            return log1p((y - this->Rrng[ip]) * this->adb[ip] / this->ada[ip]) / this->adb[ip] + this->ts[ip];
    }
};

#endif
