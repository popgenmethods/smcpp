#include "transition.h"

template <typename T>
Matrix<T> HJTransition<T>::matrix_exp(T c_rho, T c_eta)
{
    Matrix<T> Q(3, 3);
    Q << 
        exp(-2 * c_rho), (exp(-2 * c_rho) - exp(-2 * c_eta)) * c_rho / (c_eta - c_rho), this->eta->zero,
        this->eta->zero, exp(-2 * c_eta), this->eta->one - exp(-2 * c_eta),
        this->eta->zero, this->eta->zero, this->eta->one;
    Q(0, 2) = this->eta->one - Q(0, 0) - Q(0, 1);
    check_nan(Q);
    return Q;
}

template <typename T>
void HJTransition<T>::compute(void)
{
    const PiecewiseExponentialRateFunction<T> *eta = this->eta;
    std::vector<double> times;
    std::vector<Matrix<T> > expms;
    expms.push_back(Matrix<T>::Identity(3, 3));
    for (double t = 0.0; t < eta->tmax; t += delta)
    {
        T c_rho = delta * this->rho;
        T c_eta = eta->R(t + delta) - eta->R(t);
        times.push_back(t);
        expms.push_back(expms.back() * matrix_exp(c_rho, c_eta));
    }
    std::vector<Matrix<T> > expms_hs;
    expms_hs.push_back(Matrix<T>::Identity(3, 3));
    for (int k = 1; k < this->M; ++k)
    {
        int ip = insertion_point(eta->hidden_states[k], times, 0, times.size());
        T c_rho = (eta->hidden_states[k] - times[ip]) * this->rho;
        T c_eta = eta->R(eta->hidden_states[k]) - eta->R(times[ip]);
        expms_hs.push_back(expms[ip] * matrix_exp(c_rho, c_eta));
    }
    this->Phi.setZero();
    Vector<T> expms_diff(this->M - 2);
    for (int k = 1; k < this->M - 1; ++k)
        expms_diff(k - 1) = expms_hs[k](0, 2) - expms_hs[k - 1](0, 2);
    expms_diff *= 0.5;
    for (int k = 2; k < this->M; ++k)
        this->Phi.block(k - 1, 0, 1, k - 1) = expms_diff.head(k - 1).transpose();
    const int Q = 50;
#pragma omp parallel for
    for (int j = 1; j < this->M; ++j)
    {
        std::mt19937 gen;
        gen.seed(1);
        const PiecewiseExponentialRateFunction<T> myeta(*eta);
        T r, p_coal;
        std::vector<T> rtimes;
        for (int q = 0; q < Q; ++q)
            // Sample coalescence times in this interval
            rtimes.push_back(myeta.random_time(myeta.hidden_states[j - 1], myeta.hidden_states[j], gen));
        for (int k = j + 1; k < this->M; ++k)
        {
            for (int q = 0; q < Q; ++q)
            {
                p_coal = exp(-(myeta.R(eta->hidden_states[k - 1]) - myeta.R(rtimes[q])));
                if (k < this->M - 1)
                    p_coal *= -expm1(-(myeta.R(eta->hidden_states[k]) - myeta.R(eta->hidden_states[k - 1])));
                unsigned int ip = insertion_point(rtimes[q], times, 0, times.size());
                if (ip >= times.size())
                    throw std::runtime_error("erroneous insertion point");
                // this copy is to avoid some race condition that is resulting
                // in a double free.
                T tip = times[ip];
                T dt = rtimes[q] - tip;
                T c_rho = dt * this->rho;
                T c_eta = myeta.R(rtimes[q]) - myeta.R(tip);
                Matrix<T> tmp = expms[ip] * matrix_exp(c_rho, c_eta);
                r = tmp(0, 1) * p_coal;
                this->Phi(j - 1, k - 1) += r / Q;
            }
        }
        T rowsum = this->Phi.row(j - 1).sum();
        this->Phi(j - 1, j - 1) = myeta.one - rowsum;
        T thresh = 1e-20 * myeta.one;
        this->Phi = this->Phi.unaryExpr([thresh] (const T x) { if (x <= thresh) return thresh; return x; });
    }
}

template <typename T>
Matrix<T> compute_transition(const PiecewiseExponentialRateFunction<T> &eta, const T rho)
{
    DEBUG << "computing transition";
    Matrix<T> ret = HJTransition<T>(eta, rho).matrix();
    DEBUG << "done computing transition";
    return ret;
}

template Matrix<double> compute_transition(const PiecewiseExponentialRateFunction<double> &eta, const double rho);
template Matrix<adouble> compute_transition(const PiecewiseExponentialRateFunction<adouble> &eta, const adouble rho);
