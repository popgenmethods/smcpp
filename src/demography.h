#ifndef DEMOGRAPHY_H
#define DEMOGRAPHY_H

template <typename T, size_t P>
class Demography
{
    public:
    virtual PiecewiseExponentialRateFunction<T> distinguishedEta();
};

class OnePopDemography : public Demography<T, 1>
{
    public:
    OnePopDemography(ParameterVector params) : 
        eta(PiecewiseExponentialRateFunction(params)) {}

    PiecewiseExponentialRateFunction<T> distinguishedEta()
    {
        return eta;
    }

    private:
    PiecewiseExponentialRateFunction<T> eta;
};

template <typename T>
class TwoPopDemography : public Demography<T, 2>
{
    public:
    TwoPopDemography(ParameterVector params1, ParameterVector params2, double split_time) : 
        eta1(PiecewiseExponentialRateFunction(params1)),
        eta2(PiecewiseExponentialRateFunction(params2)),
        split(split) {}

    PiecewiseExponentialRateFunction<T> distinguishedEta()
    {
        return eta1;
    }

    private:
    PiecewiseExponentialRateFunction<T> eta1, eta2;
    double split;
};

#endif
