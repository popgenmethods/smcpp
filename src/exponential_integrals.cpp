#include "exponential_integrals.h"

double expintei(const double &x, const double &y)
{
    gsl_sf_result res;
    gsl_sf_expint_Ei_scaled_e(x, &res);
    return exp(x + y) * res.val;
}

template <>
double eintdiff(const double &a, const double &b, const double &c)
{
    double ret = expintei(b, c) - expintei(a, c);
    check_nan(ret);
    return ret;
}

template <>
adouble eintdiff(const adouble &a, const adouble &b, const adouble &c)
{
    adouble ret;
    ret.value() = eintdiff<double>(a.value(), b.value(), c.value());
    ret.derivatives() = exp(b.value() + c.value()) / b.value() * b.derivatives() - exp(a.value() + c.value()) / a.value() * a.derivatives();
    ret.derivatives() += c.derivatives() * ret.value();
    return ret;
}

