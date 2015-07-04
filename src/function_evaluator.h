#ifndef FUNCTION_EVALUATOR_H
#define FUNCTION_EVALUATOR_H

template <typename T>
class FunctionEvaluator 
{
    public:
    virtual ~FunctionEvaluator() = default;
    virtual std::vector<T> getTimes(void) const = 0;
    virtual T operator()(const T &x) const = 0;
    virtual std::vector<T> operator()(const std::vector<T> &x) const = 0;
    double numint(const double x, void* rate) 
    { 
        double drate = *((double*)rate);
        return exp(-drate * toDouble(this(x)));
    }
};

#endif
