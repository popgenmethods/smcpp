#ifndef RATE_FUNCTION_H
#define RATE_FUNCTION_H

#include "function_evaluator.h"

/*
 * template <typename T>
class feval : public std::unique_ptr<FunctionEvaluator<T>>
{
    T operator()(const T& t) { return *this->operator()(t); }
};
*/
template <typename T>
using feval = std::unique_ptr<FunctionEvaluator<T>>;

template <typename T>
class RateFunction
{
    public:
    RateFunction(const std::vector<std::vector<double>> &params);
    // virtual const FunctionEvaluator<T>* getEta() const = 0;
    virtual const FunctionEvaluator<T>* getR() const = 0;
    virtual const FunctionEvaluator<T>* getRinv() const = 0;
    virtual const T regularizer(void) const = 0;
    const int J, K;
    const Eigen::VectorXd z;
    const T zero;
    const T one;

    virtual void print_debug() const
    {
        std::cout << "ad_params: " << std::endl;
        for (auto v : ad_params)
            std::cout << v << std::endl << std::endl;
    }

    private:
    std::vector<std::vector<double>> params;
    void validate() 
    {
        for (auto v : params)
            if (v.size() != K)
                throw std::domain_error("Vectors must have same size");
    }
    T derivative_initializer(double);
    T derivative_initializer(double, int, int);

    protected:
    std::vector<std::vector<T>> ad_params;
};

template <typename T>
RateFunction<T>::RateFunction(const std::vector<std::vector<double>> &params) : 
    J(params.size()), K(params[0].size()), z(Eigen::VectorXd::Zero(J * K)),
    zero(derivative_initializer(0.0)), one(derivative_initializer(1.0)), 
    params(params), ad_params(J, std::vector<T>(K))
{
    validate();
    for (int j = 0; j < J; ++j)
        for (int k = 0; k < K; ++k)
            ad_params[j][k] = derivative_initializer(params[j][k], j, k);
}

template <>
inline adouble RateFunction<adouble>::derivative_initializer(double x)
{
    return adouble(x, z);
}

template <>
inline double RateFunction<double>::derivative_initializer(double x)
{
    return x;
}

template <>
inline adouble RateFunction<adouble>::derivative_initializer(double x, int j, int k)
{
    return adouble(x, J * K, j * K + k);
}

template <>
inline double RateFunction<double>::derivative_initializer(double x, int j, int k)
{
    // Silence unused-variable warnings.
    (void)j; (void)k;
    return x;
}

#endif
