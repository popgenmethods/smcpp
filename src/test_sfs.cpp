#include "common.h"
#include "conditioned_sfs.h"

int main(int argc, char** argv)
{
    const double theta = 4. * 10000. * 2.5e-8;
    std::vector<std::vector<double> > params = {
        {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
        {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
        {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}
    };
    std::vector<double> hs = {19.077194, INFINITY};
    std::vector<std::pair<int, int> > deriv = { {0,0} };
    PiecewiseExponentialRateFunction<adouble> eta(params, deriv, hs);
    const int n = 8;
    ConditionedSFS<adouble> csfs(n, 1);
    /*
    Matrix<mpreal_wrapper<T> > ts_integrals(eta.K, n - 2 + 1); 
    for (int m = 0; m < eta.K; ++m)
        eta.tjj_double_integral_below(n, csfs.mcache.prec, m, ts_integrals);
        */
    Matrix<adouble> M = csfs.compute(eta, theta)[0];
    // Matrix<adouble> M = csfs.compute_below(eta)[0];
    std::vector<std::vector<double> > params2 = {
        {1.0+1e-8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
        {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
        {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}
    };
    PiecewiseExponentialRateFunction<adouble> eta2(params2, deriv, hs);
    /*
    Matrix<mpreal_wrapper<T> > ts_integrals2(eta.K, n + 1); 
    for (int m = 0; m < eta.K; ++m)
        eta.tjj_double_integral_below(n, csfs.mcache.prec, m, ts_integrals2);
        */
    // Matrix<adouble> M2 = csfs.compute_below(eta2)[0];
    Matrix<adouble> M2 = csfs.compute(eta2, theta)[0];
    // std::cout << M.template cast<double>() << std::endl << std::endl;
    // std::cout << M2.template cast<double>() << std::endl << std::endl;
    std::cout.precision(15);
    for (int i = 0; i < M.cols(); ++i)
    {
        std::cout << i << " " << M(0,i).value() << " " << M2(0,i).value() << " " << (M2(0,i) - M(0,i)).value() / 1e-8 << " " << M(0,i).derivatives() << std::endl;
        std::cout << i << " " << M(1,i).value() << " " << M2(1,i).value() << " " << (M2(1,i) - M(1,i)).value() / 1e-8 << " " << M(1,i).derivatives() << std::endl;
        std::cout << i << " " << M(2,i).value() << " " << M2(2,i).value() << " " << (M2(2,i) - M(2,i)).value() / 1e-8 << " " << M(2,i).derivatives() << std::endl;
    }
    return 1;
    int K = eta.K;
    Matrix<mpreal_wrapper<adouble> > ts_integrals(K, n + 1); 
    Matrix<mpreal_wrapper<adouble> > ts_integrals2(K, n + 1); 
    for(int k = 0; k < K; ++k)
    {
        eta.tjj_double_integral_below(n, csfs.mcache.prec, k, ts_integrals);
        eta2.tjj_double_integral_below(n, csfs.mcache.prec, k, ts_integrals2);
        // std::cout << "*** " << k << std::endl;
        for (int i = 0; i < n + 1; ++i) 1;
            // std::cout << i << " " << (ts_integrals2(k,i) - ts_integrals(k,i)).value() / 1e-8 
                // << " " << ts_integrals(k, i).derivatives() << std::endl;
    }
    int H = 1;
    Matrix<mpreal_wrapper<adouble> > tjj_below(H, n + 1), tjj_below2(H, n + 1);
    Matrix<mpreal_wrapper<adouble> > last = ts_integrals.topRows(eta.hs_indices[0]).colwise().sum(), 
        last2 = ts_integrals2.topRows(eta.hs_indices[0]).colwise().sum(), next, next2;
    for (int h = 1; h < H + 1; ++h)
    {
        next = ts_integrals.topRows(eta.hs_indices[h]).colwise().sum();
        next2 = ts_integrals2.topRows(eta.hs_indices[h]).colwise().sum();
        tjj_below.row(h - 1) = next - last;
        tjj_below2.row(h - 1) = next2 - last2;
        last = next;
        last2 = next2;
    }
    int k = 0;
    Matrix<adouble> M0_below = csfs.below0(tjj_below), M0_below2 = csfs.below0(tjj_below2);
    Matrix<adouble> M1_below = csfs.below1(tjj_below), M1_below2 = csfs.below1(tjj_below2);
    /*
    for (int i = 0; i < tjj_below.cols(); ++i)
    {
        std::cout << i << " " << (tjj_below2(k,i) - tjj_below(k,i)).value() / 1e-8 << " " << tjj_below(k,i).derivatives() << std::endl;
    }
    for (int i = 0; i < M0_below.cols(); ++i)
    {
        std::cout << i << " " << (M0_below(0,i) - M0_below2(0,i)).value() / 1e-8 << " " << M0_below(0,i).derivatives() << std::endl;
    }
    for (int i = 0; i < M1_below.cols(); ++i)
    {
        std::cout << i << " " << (M1_below(0,i) - M1_below2(0,i)).value() / 1e-8 << " " << M1_below(0,i).derivatives() << std::endl;
    }
    */
    Matrix<adouble> ret(3, n + 1), ret2(3, n + 1);
    ret.setZero();
    ret2.setZero();
    adouble h1(0.0), h2(0.0);

    int h = 0;
    ret.block(0, 1, 1, n) = M0_below.row(h);
    ret.block(1, 0, 1, n + 1) = M1_below.row(h);
    h1 = exp(-(*(eta.getR()))(eta.hidden_states[h]));
    if (eta.hidden_states[h + 1] == INFINITY)
        h2 = 0.0;
    else
        h2 = exp(-(*(eta.getR()))(eta.hidden_states[h + 1]));
    ret /= h1 - h2;

    ret2.block(0, 1, 1, n) = M0_below2.row(h);
    ret2.block(1, 0, 1, n + 1) = M1_below2.row(h);
    h1 = exp(-(*(eta2.getR()))(eta2.hidden_states[h]));
    if (eta2.hidden_states[h + 1] == INFINITY)
        h2 = 0.0;
    else
        h2 = exp(-(*(eta2.getR()))(eta2.hidden_states[h + 1]));
    ret2 /= h1 - h2;

    Matrix<double> d(2, n + 1), d2(2, n + 1);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < n + 1; ++j)
        {
            d(i, j) = M(i,j).derivatives()(0);
            d2(i, j) = ret(i, j).derivatives()(0);
            // std::cout << i << " " << j << " " << (ret2(i,j)-ret(i,j)).value() / 1e-8 << " " << ret(i,j).derivatives() << std::endl;
            // std::cout << ret(i,j).derivatives() << " " << M(i,j).derivatives() << std::endl;
        }

    // std::cout << (M2 - M).template cast<double>()  * 1e8 << std::endl << std::endl;
    // std::cout << (ret2 - ret).template cast<double>() * 1e8 << std::endl << std::endl;
    // std::cout << (M - ret).template cast<double>()  * 1e8 << std::endl << std::endl;
    // std::cout << (M2 - ret2).template cast<double>()  * 1e8 << std::endl << std::endl;
    // std::cout << d << std::endl << std::endl;
    // std::cout << d2 << std::endl;
}

