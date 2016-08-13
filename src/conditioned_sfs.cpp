#include "conditioned_sfs.h"

template <typename T>
OnePopConditionedSFS<T>::OnePopConditionedSFS(int n) : 
    n(n),
    mei(compute_moran_eigensystem(n)), 
    mcache(cached_matrices(n)),
    Uinv_mp0(mei.Uinv.rightCols(n).template cast<double>()), 
    Uinv_mp2(mei.Uinv.reverse().leftCols(n).template cast<double>())
{}

template <typename T>
std::vector<Matrix<T> > OnePopConditionedSFS<T>::compute_below(const PiecewiseConstantRateFunction<T> &eta) const
{
    DEBUG << "compute below";
    const int M = eta.getHiddenStates().size() - 1;
    std::vector<Matrix<T> > csfs_below(M, Matrix<T>::Zero(3, n + 1));
    Matrix<T> tjj_below(M, n + 1);
    tjj_below.setZero();
    DEBUG << "tjj_double_integral below starts";
#pragma omp parallel for
    for (int m = 0; m < M; ++m)
            eta.tjj_double_integral_below(this->n, m, tjj_below);
    DEBUG << "tjj_double_integral below finished";
    DEBUG << "matrix products below (M0)";
    Matrix<T> M0_below = tjj_below * mcache.M0.template cast<T>();
    DEBUG << "matrix products below (M1)";
    Matrix<T> M1_below = tjj_below * mcache.M1.template cast<T>();
    DEBUG << "filling csfs_below";
    for (int m = 0; m < M; ++m) 
    {
        csfs_below[m].setZero();
        csfs_below[m].block(0, 1, 1, n) = M0_below.row(m);
        csfs_below[m].block(1, 0, 1, n + 1) = M1_below.row(m);
        CHECK_NAN(csfs_below[m]);
    }
    DEBUG << "compute below finished";
    return csfs_below;
}

template <typename T>
std::vector<Matrix<T> > OnePopConditionedSFS<T>::compute_above(const PiecewiseConstantRateFunction<T> &eta) const
{
    const int M = eta.getHiddenStates().size() - 1;
    std::vector<Matrix<T> > C_above(M, Matrix<T>::Zero(n + 1, n)), 
        csfs_above(M, Matrix<T>::Zero(3, n + 1));
    DEBUG << "compute above";
#pragma omp parallel for
    for (int j = 2; j < n + 3; ++j)
        eta.tjj_double_integral_above(n, j, C_above);
    Matrix<T> tmp;

#pragma omp parallel for
    for (int m = 0; m < M; ++m)
    {
        csfs_above[m].setZero();
        Matrix<T> C0 = C_above[m].transpose(), C2 = C_above[m].colwise().reverse().transpose();
        Vector<T> tmp0(this->mcache.X0.cols()), tmp2(this->mcache.X2.cols());
        tmp0.setZero();
        for (int j = 0; j < this->mcache.X0.cols(); ++j)
        {
            std::vector<T> v;
            for (int i = 0; i < this->mcache.X0.rows(); ++i)
                v.push_back(this->mcache.X0(i, j) * C0(i, j));
            std::sort(v.begin(), v.end(), [] (T x, T y) { return std::abs(toDouble(x)) > std::abs(toDouble(y)); });
            tmp0(j) = doubly_compensated_summation(v);
        }
        csfs_above[m].block(0, 1, 1, n) = tmp0.transpose().lazyProduct(Uinv_mp0);
        tmp2.setZero();
        for (int j = 0; j < this->mcache.X2.cols(); ++j)
        {
            std::vector<T> v;
            for (int i = 0; i < this->mcache.X2.rows(); ++i)
                v.push_back(this->mcache.X2(i, j) * C2(i, j));
            std::sort(v.begin(), v.end(), [] (T x, T y) { return std::abs(toDouble(x)) > std::abs(toDouble(y)); });
            tmp2(j) = doubly_compensated_summation(v);
        }
        csfs_above[m].block(2, 0, 1, n) = tmp2.transpose().lazyProduct(Uinv_mp2);
        CHECK_NAN(csfs_above[m]);
    }
    return csfs_above;
}

template <typename T>
std::vector<Matrix<T> > OnePopConditionedSFS<T>::compute(const PiecewiseConstantRateFunction<T> &eta) const
{
    DEBUG << "compute called";
    const int M = eta.getHiddenStates().size() - 1;
    std::vector<Matrix<T> > csfs_above = compute_above(eta);
    std::vector<Matrix<T> > csfs_below = compute_below(eta);
    std::vector<Matrix<T> > csfs(M, Matrix<T>::Zero(3, n + 1));
    for (int m = 0; m < M; ++m)
        csfs[m] = csfs_above[m] + csfs_below[m];
    DEBUG << "compute finished";
    return csfs;
}

template <typename T>
std::vector<Matrix<T> > incorporate_theta(const std::vector<Matrix<T> > &csfs, double theta)
{
    std::vector<Matrix<T> > ret(csfs.size());
    for (unsigned int i = 0; i < csfs.size(); ++i)
    {
        T tauh = csfs[i].sum();
        if (toDouble(tauh) > 1.0 / theta)
            throw improper_sfs_exception();
        try
        {
            CHECK_NAN(tauh);
            ret[i] = csfs[i] * -expm1(-theta * tauh) / tauh;
            CHECK_NAN(ret[i]);
        } catch (std::runtime_error)
        {
            std::cout << i << std::endl << csfs[i].template cast<double>() << std::endl;
            std::cout << tauh << std::endl;
            std::cout << theta << std::endl;
            throw;
        }
        T tiny = tauh - tauh + 1e-20;
        ret[i] = ret[i].unaryExpr([=](const T x) { if (x < 1e-20) return tiny; if (x < -1e-8) throw std::domain_error("very negative sfs"); return x; });
        CHECK_NAN(ret[i]);
        tauh = ret[i].sum();
        ret[i](0, 0) = 1. - tauh;
        try { CHECK_NAN(ret[i]); }
        catch (std::runtime_error)
        {
            std::cout << i << "\n" << ret[i].template cast<double>() << std::endl;
            throw;
        }
        if (ret[i].template cast<double>().minCoeff() < 0 or ret[i].template cast<double>().maxCoeff() > 1)
        {
            std::cout << i << std::endl << csfs[i].template cast<double>() << std::endl;
            throw std::runtime_error("csfs is not a probability distribution");
        }
    }
    return ret;
}

template std::vector<Matrix<double> > incorporate_theta(const std::vector<Matrix<double> > &csfs, double theta);
template std::vector<Matrix<adouble> > incorporate_theta(const std::vector<Matrix<adouble> > &csfs, double theta);

template class OnePopConditionedSFS<double>;
template class OnePopConditionedSFS<adouble>;
