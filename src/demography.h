#ifndef DEMOGRAPHY_H
#define DEMOGRAPHY_H

template <typename T>
class Demography
{
    public:
    virtual ~Demography() = default;
    virtual PiecewiseConstantRateFunction<T> distinguishedEta() const = 0;
};

template <typename T>
class OnePopDemography : public Demography<T>
{
    public:
    OnePopDemography(ParameterVector params, std::vector<double> hidden_states) : 
        eta(PiecewiseConstantRateFunction<T>(params, hidden_states)) {}

    PiecewiseConstantRateFunction<T> distinguishedEta() const
    {
        return eta;
    }

    private:
    const PiecewiseConstantRateFunction<T> eta;
};

template <typename T>
class TwoPopDemography : public Demography<T>
{
    public:
    TwoPopDemography(ParameterVector params1, ParameterVector params2, double split_time) : 
        eta1(PiecewiseConstantRateFunction<T>(params1)),
        eta2(PiecewiseConstantRateFunction<T>(params2)),
        split(split) {}

    PiecewiseConstantRateFunction<T> distinguishedEta()
    {
        return eta1;
    }

    private:
    PiecewiseConstantRateFunction<T> eta1, eta2;
    double split;
};

#endif
