#ifndef FUNCTION_EVALUATOR_H
#define FUNCTION_EVALUATOR_H

template <typename T>
class FunctionEvaluator 
{
    public:
    virtual T operator()(const T &x) const = 0;
    virtual std::vector<T> operator()(const std::vector<T> &x) const = 0;
};

#endif
