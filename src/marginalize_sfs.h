#ifndef MARGINALIZE_SFS_H
#define MARGINALIZE_SFS_H

#include "gsl/gsl_randist.h"

template <size_t P>
struct marginalize_sfs
{
    template <typename Derived1, typename Derived2, typename Derived3, typename Derived4>
    Vector<typename Derived1::Scalar> operator()(
        const Eigen::MatrixBase<Derived1> &sfs, 
        const Eigen::MatrixBase<Derived2> &n, 
        const Eigen::MatrixBase<Derived3> &b, 
        const Eigen::MatrixBase<Derived4> &nb);
};

template <>
struct marginalize_sfs<1>
{
    template<typename Derived1, typename Derived2, typename Derived3, typename Derived4>
    Vector<typename Derived1::Scalar> operator()(
        const Eigen::MatrixBase<Derived1> &sfs, 
        const Eigen::MatrixBase<Derived2> &n, 
        const Eigen::MatrixBase<Derived3> &b, 
        const Eigen::MatrixBase<Derived4> &nb)
    {
        typedef typename Derived1::Scalar T;
        int M = sfs.rows();
        Vector<T> ret = Vector<T>::Zero(M);
        assert(sfs.cols() == n(0) + 1);
        for (unsigned int n1 = b(0); n1 < n(0) + b(0) - nb(0) + 1; ++n1)
        {
            // n1: number of derived in sample of size n
            // n2: number of ancestral "    "   "   "
            // must have: b(0) <= n1, nb(0) - b(0) <= n2 = n - n1 => n1 <= n + b - nb
            unsigned int n2 = n(0) - n1;
            // p(k) =  C(n_1, k) C(n_2, t - k) / C(n_1 + n_2, t)
            // gsl_ran_hypergeometric_pdf(unsigned int k, unsigned int n1, unsigned int n2, unsigned int t)
            ret += gsl_ran_hypergeometric_pdf(b(0), n1, n2, nb(0)) * sfs.col(n1);
        }
        return ret;
    }
};

template <size_t P>
template <typename Derived1, typename Derived2, typename Derived3, typename Derived4>
Vector<typename Derived1::Scalar> marginalize_sfs<P>::operator()(
        const Eigen::MatrixBase<Derived1> &sfs, 
        const Eigen::MatrixBase<Derived2> &n, 
        const Eigen::MatrixBase<Derived3> &b, 
        const Eigen::MatrixBase<Derived4> &nb)
{
    typedef typename Derived1::Scalar T;
    assert(sfs.cols() == (n.array() + 1).prod());
    int M = sfs.rows();
    Vector<T> ret = Vector<T>::Zero(M);
    Matrix<T> marginal_sfs = Matrix<T>::Zero(M, n(0) + 1);
    int tl = n.size() - 1;
    int slice = (n.tail(tl).array() + 1).prod();
    assert((n(0) + 1) * slice == sfs.cols());
    for (int i = 0; i < n(0) + 1; ++i)
        marginal_sfs.col(i) = marginalize_sfs<P - 1>()(
                sfs.middleCols(i * slice, slice), 
                n.tail(tl), b.tail(tl), nb.tail(tl));
    return marginalize_sfs<1>()(marginal_sfs, n.head(1), b.head(1), nb.head(1));
}

#endif
