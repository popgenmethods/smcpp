#include "transition.h"

namespace hj_transition
{
    const double A_rho_data[] = {
        -1, 1, 0,
         0, 0, 0,
         0, 0, 0};
    static Eigen::Matrix<double, 3, 3, Eigen::RowMajor> A_rho(A_rho_data);

    const double A_eta_data[] = {
        0, 0, 0,
        1, -2, 1,
        0, 0, 0};
    static Eigen::Matrix<double, 3, 3, Eigen::RowMajor> A_eta(A_eta_data);

    Matrix<double> transition_exp(double c_rho, double c_eta)
    {
        Matrix<double> M = c_rho * A_rho + c_eta * A_eta;
        return M.exp();
    }

    struct expm_functor
    {
        typedef double Scalar;
        typedef Eigen::Matrix<Scalar, 1, 2> InputType;
        typedef Eigen::Matrix<Scalar, 9, 1> ValueType;
        typedef Eigen::Matrix<Scalar, 9, 2> JacobianType;

        static const int InputsAtCompileTime = 1;
        static const int ValuesAtCompileTime = 9;

        static int values() { return 9; }

        expm_functor() {}
        int operator()(const InputType &x, ValueType &f) const
        {
            Eigen::Matrix<double, 3, 3, Eigen::ColMajor> M = transition_exp(x(0), x(1));
            f = Eigen::Matrix<double, 9, 1, Eigen::ColMajor>::Map(M.data(), 9, 1);
            return 0;
        }
    };

    Matrix<adouble> transition_exp(adouble c_rho, adouble c_eta)
    {
        // Compute derivative dependence on c_eta by numerical differentiation
        expm_functor f;
        Eigen::NumericalDiff<expm_functor> numDiff(f);
        Eigen::Matrix<double, 9, 2> df;
        Eigen::Matrix<double, 1, 2> meta;
        meta(0, 0) = c_rho.value();
        meta(0, 1) = c_eta.value();
        numDiff.df(meta, df);
        Matrix<adouble> ret = transition_exp(c_rho.value(), c_eta.value()).cast<adouble>();
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                ret(i, j).derivatives() = c_rho.derivatives() * df(3 * j + i, 0) + 
                    c_eta.derivatives() * df(3 * j + i, 1);
        return ret;
    }
};

template <typename T>
Matrix<T> matexp3x3(T crho, T ceta)
{
    Matrix<T> cg0(3, 3);
    T t1 = (crho * crho);
    T t2 = (ceta * ceta);
    T t3 = t1 + 4 * t2;
    T t4 = 1 / sqrt(t3);
    t3 *= t4;
    T t5 = -t3 / 0.2e1 - crho / 0.2e1 - ceta;
    T t6 = exp(t5);
    T t7 = t3 / 0.2e1 - crho / 0.2e1 - ceta;
    T t8 = exp(t7);
    T t9 =  t3 + crho - 0.2e1 * ceta;
    T t10 = 0.2e1 * ceta - crho +  t3;
    t5 *= -0.2e1;
    t7 *= 0.2e1;
    T t11 = t6 - t8;
    t7 = 0.1e1 / t7;
    t5 = 0.1e1 / t5;
    T t12 = ceta * crho;
    T t13 = t11 * crho;
    T zero = 0 * ceta;
    T one = zero + 1;
    cg0(0,0) = -0.2e1 * t12 * (t10 * t8 + t6 * t9) * t4 * t5 * t7;
    cg0(0,1) = 0.4e1 *  t1 * ceta * t11 * t4 * t5 * t7;
    cg0(0,2) = one - cg0(0,0) - cg0(0,1);
    cg0(1,0) = 0.4e1 * t13 *  t2 * t4 * t5 * t7;
    cg0(1,1) = -0.2e1 * t12 * (t10 * t6 + t8 * t9) * t4 * t5 * t7;
    cg0(1,2) = one - cg0(1,0) - cg0(1,1);
    cg0(2,0) = zero;
    cg0(2,1) = zero;
    cg0(2,2) = one;
    return cg0;
}

void print_derivatives(double) {}
void print_derivatives(adouble x) {
    std::cout << x.derivatives().transpose() << std::endl;
}

template <typename T>
void HJTransition<T>::compute(void)
{
    using namespace hj_transition;
    std::mt19937 gen;
    gen.seed(1);
    const PiecewiseExponentialRateFunction<T>* eta = this->eta;
    auto R = eta->getR();
    T r, p_coal;
    this->Phi.setZero();
    const int Q = 10;
    for (int j = 1; j < this->M; ++j)
    {
        std::vector<std::pair<T, T> > rtimes;
        for (int q = 0; q < Q; ++q)
        {
            T rand_time = eta->random_time(eta->hidden_states[j - 1], eta->hidden_states[j], gen);
            rtimes.emplace_back(eta->zero + 2 * this->rho * rand_time, (*R)(rand_time));
        }
        // Sample coalescence times in this interval
        for (int k = 1; k < j; ++k)
            this->Phi(j - 1, k - 1) = expm(0, k)(0, 2) - expm(0, k - 1)(0, 2);
//         if (j == this->M - 1)
//             this->Phi(j - 1, j - 1) = 1. - expm(0, j - 1)(0, 2);
//         else
//         {
//             // for (int q = 0; q < Q; ++q)
//                 // this->Phi(j - 1, j - 1) += matexp3x3(0, 0, rtimes[q].first, rtimes[q].second) / Q;
//             this->Phi(j - 1, j - 1) = expm(0, j)(0, 0);
//             for (int i = 0; i < 2; ++i)
//                 this->Phi(j - 1, j - 1) += expm(0, j - 1)(0, i) * expm(j - 1, j)(i, 2);
//         }
        for (int k = j + 1; k < this->M; ++k)
        {
            for (int q = 0; q < Q; ++q)
            {
                T c_rho = rtimes[q].first;
                T c_eta = rtimes[q].second;
                p_coal = exp(-((*R)(eta->hidden_states[k - 1]) - c_eta));
                if (k < this->M - 1)
                    p_coal *= -expm1(-((*R)(eta->hidden_states[k]) - (*R)(eta->hidden_states[k - 1])));
                r = transition_exp(c_rho, c_eta)(0, 1);
                r *= p_coal;
                this->Phi(j - 1, k - 1) += r / Q;
            }
        }
        this->Phi(j - 1, j - 1) = 0;
        T rowsum = this->Phi.row(j - 1).sum();
        this->Phi(j - 1, j - 1) = eta->one - rowsum;
    }
}

template <typename T>
Matrix<T> HJTransition<T>::expm(int i, int j)
{
    using namespace hj_transition;
    const PiecewiseExponentialRateFunction<T> *eta = this->eta;
    auto R = eta->getR();
    const int M = this->M;
    T c_rho = eta->zero + 2 * this->rho * (eta->hidden_states[j] - eta->hidden_states[i]);
    T c_eta = (*R)(eta->hidden_states[j]) - (*R)(eta->hidden_states[i]);
    Matrix<T> ret(3, 3);
    if (i == j)
        ret = Matrix<T>::Identity(3, 3);
    else
        ret = transition_exp(c_rho, c_eta);
    return ret;
}

template <typename T>
Matrix<T> compute_transition(const PiecewiseExponentialRateFunction<T> &eta, const double rho, bool useHJ)
{
    return HJTransition<T>(eta, rho).matrix();
}

template Matrix<double> compute_transition(const PiecewiseExponentialRateFunction<double> &eta, const double rho, bool);
template Matrix<adouble> compute_transition(const PiecewiseExponentialRateFunction<adouble> &eta, const double rho, bool);

int _main(int argc, char** argv)
{
    using namespace hj_transition;
    adouble c_eta(30.0, 3, 0); 
    adouble c_rho(.01, 3, 1);
    adouble c_eta2(30.0 + 1e-8, 3, 0); 
    adouble c_rho2(.01 + 1e-8, 3, 1);
    Matrix<adouble> me = transition_exp(c_rho, c_eta);
    Matrix<adouble> me2 = transition_exp(c_rho, c_eta2);
    Matrix<adouble> me3 = transition_exp(c_rho2, c_eta);
    Matrix<adouble> mme =  matexp3x3(c_rho, c_eta);
    Matrix<adouble> mme2 = matexp3x3(c_rho, c_eta2);
    Matrix<adouble> mme3 = matexp3x3(c_rho2, c_eta);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            std::cout << i << "," << j << " " << 
                me(i,j).value() << " " << mme(i, j).value() << std::endl <<
                me(i,j).derivatives().transpose() << " :: ";
            std::cout << (me2(i,j) - me(i,j)) / 1e-8 << " ";
            std::cout << (me3(i,j) - me(i,j)) / 1e-8 << std::endl;
            std::cout << mme(i,j).derivatives().transpose() << " :: ";
            std::cout << (mme2(i,j) - mme(i,j)) / 1e-8 << " ";
            std::cout << (mme3(i,j) - mme(i,j)) / 1e-8 << std::endl << std::endl;
        }
    return 0;
    std::vector<std::vector<double> > params = {
        {0.2, 1.01, 2.0},
        {1.0, 1.0, 2.0},
        {0.1, 0.1, 0.1}
    };
    std::vector<double> hs;
    int Q = 50;
    for (int i = 0; i < Q + 1; ++i)
        hs.push_back(2 * (double)i / Q);
    std::cout << hs << std::endl;
    std::vector<std::pair<int, int> > deriv;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            deriv.emplace_back(i, j);
    std::cout << deriv << std::endl;
    PiecewiseExponentialRateFunction<adouble> eta(params, deriv, hs);
    Matrix<adouble> T = compute_transition(eta, .0001, true);
    std::cout << T.template cast<double>() << std::endl << std::endl;
    for (int i = 0; i < Q; ++i)
        for (int j = 0; j < Q; ++j)
            for (int k = 0; k < 2; ++k)
                for (int ell = 0; ell < 3; ++ell)
                {
                    std::vector<std::vector<double> > params2 = {
                        {0.2, 1.01, 2.0},
                        {1.0, 1.0, 2.0},
                        {0.1, 0.1, 0.1}
                    };
                    params2[k][ell] += 1e-8;
                    PiecewiseExponentialRateFunction<double> eta2(params2, deriv, hs);
                    Matrix<double> T2 = compute_transition(eta2, .0001, true);
                    std::cout << i << " " << j << " " << " " << k << " " << ell << " " << 
                        (T2(i,j) - T(i,j).value()) / 1e-8 << " " << T(i,j).derivatives()(3 * k + ell) << std::endl;
                }
}
