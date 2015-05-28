#ifndef SPLINE_RATE_FUNCTION
#define SPLINE_RATE_FUNCTION

#include "common.h"
#include "rate_function.h"
#include "piecewise_polynomial.h"

template <typename T>
class SplineRateFunction : public RateFunction<T>
{
    public:
    SplineRateFunction(const std::vector<std::vector<double>> &params);
    virtual const FunctionEvaluator<T>* getEta() const { return &eta; }
    virtual const FunctionEvaluator<T>* getR() const { return &R; }
    virtual const FunctionEvaluator<T>* getRinv() const { return &Rinv; }

    virtual void print_debug() const
    {
        RateFunction<T>::print_debug();
        std::cout << "eta: " << eta << std::endl << std::endl;
        std::cout << "R: " << R << std::endl << std::endl;
        std::cout << "Rinv: " << Rinv << std::endl << std::endl;
        std::cout << "L_2(eta''): " << this->regularizer() << std::endl;
    }

    private:
    PiecewisePolynomial<T, 1> make_inverse();
    PiecewisePolynomial<T, 6> eta;
    PiecewisePolynomial<T, 7> R;
    PiecewisePolynomial<T, 1> Rinv;
};

#endif

