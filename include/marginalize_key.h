#ifndef MARGINALIZE_SFS_H
#define MARGINALIZE_SFS_H

#include "gsl/gsl_randist.h"

template <size_t P>
struct marginalize_sfs_b;

template <size_t P>
struct marginalize_sfs_a
{
    template <typename Derived1, typename Derived2, typename Derived3,
              typename Derived4, typename Derived5, typename Derived6>
    Vector<typename Derived1::Scalar> operator()(
            const Eigen::MatrixBase<Derived1> &sfs,
            const Eigen::MatrixBase<Derived2> &n,
            const Eigen::MatrixBase<Derived3> &a,
            const Eigen::MatrixBase<Derived4> &na,
            const Eigen::MatrixBase<Derived5> &b,
            const Eigen::MatrixBase<Derived6> &nb)
    {
        typedef typename Derived1::Scalar T;
        int M = sfs.rows();
        int slice = (n.array() + 1).prod();
        slice *= (a.tail(P - 1).array() + 1).prod();
        Vector<T> ret = Vector<T>::Zero(M);
        std::vector<int> is;
        if (a(0) == -1)
            for (int aa = 0; aa < na(0) + 1; ++aa)
                is.push_back(aa);
        else
            is = {a(0)};
        for (int i : is)
        {
            ret += marginalize_sfs_b<P>()(
                    sfs.middleCols(i * slice, slice),
                    n, a.tail(P - 1), na.tail(P - 1),
                    b, nb);
        }
        return ret;
    }
};

template <size_t P>
struct marginalize_sfs_b
{
    template <typename Derived1, typename Derived2, typename Derived3,
              typename Derived4, typename Derived5, typename Derived6>
    Vector<typename Derived1::Scalar> operator()(
        const Eigen::MatrixBase<Derived1> &sfs, 
        const Eigen::MatrixBase<Derived2> &n,
        const Eigen::MatrixBase<Derived3> &a,
        const Eigen::MatrixBase<Derived4> &na,
        const Eigen::MatrixBase<Derived5> &b,
        const Eigen::MatrixBase<Derived6> &nb)
    {
        typedef typename Derived1::Scalar T;
        const int M = sfs.rows();
        const int n0 = n(0);
        int tl = n.size() - 1;
        int slice = (n.tail(tl).array() + 1).prod();
        slice *= (a.array() + 1).prod();
        Vector<T> ret = Vector<T>::Zero(M);
        Matrix<T> marginal_sfs = Matrix<T>::Zero(M, n0 + 1);
        assert((n0 + 1) * slice == sfs.cols());
        marginalize_sfs_a<P - 1> ma;
        for (int i = 0; i < n0 + 1; ++i)
            marginal_sfs.col(i) = ma(
                    sfs.middleCols(i * slice, slice),
                    n.tail(tl), a, na, b.tail(tl), nb.tail(tl));
        return marginalize_sfs_b<1>()(marginal_sfs, n.head(1), a, na, b.head(1), nb.head(1));
    }
};

template <>
struct marginalize_sfs_b<1>
{
    template <typename Derived1, typename Derived2, typename Derived3,
              typename Derived4, typename Derived5, typename Derived6>
    Vector<typename Derived1::Scalar> operator()(
        const Eigen::MatrixBase<Derived1> &sfs, 
        const Eigen::MatrixBase<Derived2> &n,
        const Eigen::MatrixBase<Derived3> &a,
        const Eigen::MatrixBase<Derived4> &na,
        const Eigen::MatrixBase<Derived5> &b,
        const Eigen::MatrixBase<Derived6> &nb)
    {
        typedef typename Derived1::Scalar T;
        int M = sfs.rows();
        Vector<T> ret = Vector<T>::Zero(M);
        int n0 = n(0);
        assert(sfs.cols() == n0 + 1);
        assert(b.size() == 1);
        assert(nb.size() == 1);
        for (int n1 = b(0); n1 < n0 + b(0) - nb(0) + 1; ++n1)
        {
            // n1: number of derived in sample of size n
            // n2: number of ancestral "    "   "   "
            // must have: b(0) <= n1, nb(0) - b(0) <= n2 = n - n1 => n1 <= n + b - nb
            int n2 = n0 - n1;
            // p(k) =  C(n_1, k) C(n_2, t - k) / C(n_1 + n_2, t)
            // gsl_ran_hypergeometric_pdf(unsigned int k, unsigned int n1, unsigned int n2, unsigned int t)
            Vector<T> c = sfs.col(n1);
            ret += gsl_ran_hypergeometric_pdf(b(0), n1, n2, nb(0)) * c;
        }
        return ret;
    }
};

#endif
