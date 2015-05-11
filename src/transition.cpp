#include "transition.h"

const double A_rho_data[] = {
    -1, 1, 0, 0,
     0, -.5, .5, 0,
     0, 0, 0, 0,
     0, 0, 0, 0};
static Eigen::Matrix<double, 4, 4, Eigen::RowMajor> A_rho(A_rho_data);

const double A_eta_data[] = {
    0, 0, 0, 0,
    1, -2, 0, 1,
    0, 4, -5, 1,
    0, 0, 0, 0};
static Eigen::Matrix<double, 4, 4, Eigen::RowMajor> A_eta(A_eta_data);

std::array<adouble, 3> eigenvalues(double c_rho, adouble c_eta)
{
    adouble a = 7 * c_eta + (3 * c_rho) / 2.0;
    adouble b = 10 * pow(c_eta, 2) + (13 * c_eta * c_rho)/2 + pow(c_rho, 2) / 2.0;
    adouble c = 5 * pow(c_eta, 2) * c_rho + (c_eta * pow(c_rho, 2)) / 2.0;
    adouble Q = (pow(a, 2) - 3 * b) / 9.0;
    adouble R = (2 * pow(a, 3) - 9 * a * b + 27 * c) / 54.0;
    adouble theta = acos(R / pow(Q, 1.5));
    double x[3] = {0, 2 * M_PI, -2 * M_PI};
    std::array<adouble, 3> ret;
    for (int i = 0; i < 3; ++i)
    {
        ret[i] = -(2 * pow(Q, 0.5) * cos((theta + x[i])/ 3)) - a / 3;
    }
    return ret;
}

AdMatrix right_eigenvectors(double c_rho, adouble c_eta)
{
    AdMatrix ret(4, 4);
    adouble zero = 0, one = 1;
    zero.derivatives() = Eigen::VectorXd::Zero(c_eta.derivatives().rows());
    one.derivatives() = Eigen::VectorXd::Zero(c_eta.derivatives().rows());
    ret.col(0) << one, one, one, one;
    adouble r;
    std::array<adouble, 3> eigvals = eigenvalues(c_rho, c_eta);
    for (int i = 1; i < 4; ++i)
    {
        r = eigvals[i - 1];
        ret.col(i) << -c_rho/(2.*c_eta) + ((-c_rho - 4*c_eta - 2*r)*(-5*c_eta - r))/(8.*pow(c_eta,2)),-(-5*c_eta - r)/(4.*c_eta), one, zero;
    }
    return ret;
}

AdMatrix left_eigenvectors(double c_rho, adouble c_eta)
{
    AdMatrix ret(4, 4);
    // Need to make sure that derivative vectors match up here or we'll get
    // crashes later.
    adouble zero = 0, one = 1;
    zero.derivatives() = Eigen::VectorXd::Zero(c_eta.derivatives().rows());
    one.derivatives() = Eigen::VectorXd::Zero(c_eta.derivatives().rows());
    ret.row(0) << zero, zero, zero, one;
    adouble r;
    std::array<adouble, 3> eigvals = eigenvalues(c_rho, c_eta);
    for (int i = 1; i < 4; ++i)
    {
        r = eigvals[i - 1];
        ret.row(i) << (r*(c_rho*c_eta + 20*pow(c_eta,2) + c_rho*r+ 14*c_eta*r + 2*pow(r,2)))/
                (c_rho*c_eta*(c_rho + 10*c_eta + 2*r)),(2*r*(5*c_eta + r))/(c_eta*(c_rho + 10*c_eta + 2*r)),
                (c_rho*r)/(c_eta*(c_rho + 10*c_eta + 2*r)), one;
    }
    return ret;

}

std::array<AdMatrix, 3> eigensystem(double c_rho, adouble c_eta)
{
    AdMatrix L = left_eigenvectors(c_rho, c_eta);
    AdMatrix R = right_eigenvectors(c_rho, c_eta);
    AdMatrix Rinv(4, 4);
    Rinv.setZero();
    AdVector d = (L * R).diagonal();
    for (int i = 0; i < 4; ++i)
        Rinv.diagonal()(i) = 1. / d(i);
    Rinv *= L;
    std::array<adouble, 3> eig = eigenvalues(c_rho, c_eta);
    AdMatrix D(4, 4);
    D.setZero();
    D(0,0) = 0;
    for (int i = 1; i < 4; ++i)
        D(i,i) = eig[i - 1];
    return {R, D, Rinv};
}

AdMatrix transition_exp(double c_rho, adouble c_eta)
{
    std::array<AdMatrix, 3> eig = eigensystem(c_rho, c_eta);
    AdMatrix P = eig[0];
    AdMatrix D = eig[1];
    AdMatrix Pinv = eig[2];
    for (int i = 0; i < 4; ++i)
        D(i, i) = exp(D(i, i));
    AdMatrix ret = P * D * Pinv;
    /*
    Eigen::Matrix4d check = c_rho * A_rho + c_eta.value() * A_eta;
    Eigen::Matrix4d expcheck = check.exp();
    Eigen::Matrix4d ret_check = ret.cast<double>();
    std::cout << "check" 
        << std::endl << check << std::endl << "expcheck"
        << std::endl << expcheck << std::endl 
        << "ret_check" << std::endl << ret_check << std::endl << std::endl;
        */
    return ret;
}

Transition::Transition(PiecewiseExponential *eta, 
        const std::vector<double> &hidden_states, 
        double rho) :
    eta(eta), _hs(hidden_states), rho(rho), M(hidden_states.size()), I(M, M), Phi(M - 1, M - 1)
{
    I.setIdentity();
    Phi.setZero();
    compute();
}

void Transition::compute(void)
{
    adouble r, p_coal;
    for (int j = 1; j < M; ++j)
        for (int k = 1; k < M; ++k)
        {
            if (k < j)
            {
                r = expm(0, k)(0, 3) - expm(0, k - 1)(0, 3);
            }
            else if (k == j && j == M - 1)
            {
                r = 1. - expm(0, k - 1)(0, 3);
            }
            else if (k == j)
            {
                r = expm(0, k)(0, 0);
                for (auto i : {0, 1, 2})
                    r += expm(0, k - 1)(0, i) * expm(k - 1, k)(i, 3);
            }
            else
            {
                p_coal = exp(-(eta->R(_hs[k - 1]) - eta->R(_hs[j])));
                if (k < M - 1)
                {
                    // Else d[k] = +inf, coalescence in [d[k-1], +oo) is assured.
                    p_coal *= -expm1(-(eta->R(_hs[k]) - eta->R(_hs[k - 1])));
                }
                r = expm(0, j).block(0, 1, 1, 2).sum() * p_coal;
            }
        Phi(j - 1, k - 1) = r;
        }
}

AdMatrix& Transition::matrix(void) { return Phi; }

void Transition::store_results(double* outtrans, double* outjac)
{
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> _trans(outtrans, M - 1, M - 1);
    _trans = Phi.cast<double>();
    int m = 0;
    for (int i = 0; i < M - 1; ++i)
        for (int j = 0; j < M - 1; ++j)
        {
            Eigen::VectorXd d = Phi(i, j).derivatives();
            // printf("i:%i j:%i dtransition/da[0]:%f\n", i, j, d(0));
            for (int k = 0; k < d.rows(); ++k)
                outjac[m++] = d(k);
        }
}

AdMatrix Transition::expm(int i, int j)
{
    std::pair<int, int> key = {i, j};
    if (_expm_memo.count(key) == 0)
    {
        double c_rho;
        adouble c_eta;
        AdMatrix ret(M, M);
        if (i == j)
            ret = I;
        else
        {
            c_rho = rho * (_hs[j] - _hs[i]);
            c_eta = eta->R(_hs[j]) - eta->R(_hs[i]);
            /*
            AdMatrix A = c_rho * A_rho.cast<adouble>() + c_eta * A_eta.cast<adouble>();
            Eigen::HouseholderQR<AdMatrix> qr(A);
            AdMatrix Q = qr.householderQ();
            AdMatrix R = qr.matrixQR().block(0,0,A.cols(),A.cols()).triangularView<Eigen::Upper>();
            std::cout << Q.cast<double>() << std::endl;
            std::cout << R.cast<double>() << std::endl;
            std::cout << (Q * R).cast<double>() << std::endl;
            std::cout << A.cast<double>() << std::endl;
            */
            ret = transition_exp(c_rho, c_eta);
        }
        _expm_memo[key] = ret;
    }
    return _expm_memo[key];
}

/*
int main(int argc, char** argv)
{
    PiecewiseExponential pe({1.0, 2.0, 3.0}, {-0.01, .001, .03}, {0.0, 0.5, 1.0});
    std::cout << pe.R(1.0) << std::endl;
    std::array<AdMatrix, 3> eig = eigensystem(1.0, pe.R(1.0));
    AdMatrix P = eig[0];
    AdMatrix D = eig[1];
    AdMatrix Pinv = eig[2];
    Eigen::Matrix4d A; 
    AdMatrix Ap = (P * D * Pinv);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            A(i,j) = Ap(i,j).value();
    std::cout << A << std::endl << std::endl;

    for (auto x : eigenvalues(1.0, 2.0))
        std::cout << x.value() << " ";
    std::cout << std::endl;
    std::vector<double> hs = {0.0, 1.0, 2.0, 3.0, INFINITY};
    Transition T(&pe, hs, 1e-8);
    T.compute();
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            A(i,j) = Ap(i,j).value();
    std::cout << A << std::endl;
}
*/
