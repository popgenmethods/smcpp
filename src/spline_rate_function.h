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
    // virtual const feval& getEta() const { return &eta; }
    virtual const FunctionEvaluator<T>* getR() const { return R.get(); }
    virtual const FunctionEvaluator<T>* getRinv() const { return Rinv.get(); }
    virtual const T regularizer(void) const { return _reg; }

    virtual void print_debug() const
    {
        RateFunction<T>::print_debug();
        /*
         * std::cout << "R: " << *R << std::endl << std::endl;
        std::cout << "Rinv: " << *Rinv << std::endl << std::endl;
        std::cout << "L_2(eta''): " << this->regularizer() << std::endl;
        */
    }

    private:
    PiecewisePolynomial<T, 1> make_inverse();
    T _reg;
    feval<T> R, Rinv;
};

#endif

