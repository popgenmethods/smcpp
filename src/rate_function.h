#ifndef RATE_FUNCTION_H
#define RATE_FUNCTION_H

template <typename T>
class RateFunction
{
    public:
    RateFunction(const std::vector<std::vector<double>> &params);
    T regularizer(void) const { return _reg; }
    virtual T R(const T &x) const = 0;
    virtual std::vector<T> Rv(const std::vector<T> &v) const = 0;
    virtual T Rinv(const T &y, const T &x) const = 0;
    const int J, K;
    const T zero;
    const T one;

    protected:
    std::vector<std::vector<T>> ad_params;
    T _reg;

    private:
    std::vector<std::vector<double>> params;
    Eigen::VectorXd z;
    void validate() 
    {
        for (auto v : params)
            if (v.size() != J)
                throw std::domain_error("Vectors must have same size");
    }
    T derivative_initializer(double);
    T derivative_initializer(double, int, int);
};

template <typename T>
RateFunction<T>::RateFunction(const std::vector<std::vector<double>> &params) : 
    J(params.size()), K(params[0].size()), zero(derivative_initializer(0.0)), 
    one(derivative_initializer(1.0)), params(params), ad_params(J, std::vector<T>(K)), 
    z(J * K)
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
    return x;
}

#endif
