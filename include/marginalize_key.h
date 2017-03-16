#ifndef MARGINALIZE_KEY_H
#define MARGINALIZE_KEY_H

#include <map>
#include "gsl/gsl_randist.h"

#include "block_key.h"

template <size_t P>
struct marginalize_key
{
    template <typename Derived1, typename Derived2, typename Derived3>
    static std::map<block_key, double> run(
        const Eigen::MatrixBase<Derived1> &key,
        const Eigen::MatrixBase<Derived2> &n,
        const Eigen::MatrixBase<Derived3> &na);
};

template <>
template <typename Derived1, typename Derived2, typename Derived3>
std::map<block_key, double> marginalize_key<1>::run(
    const Eigen::MatrixBase<Derived1> &key,
    const Eigen::MatrixBase<Derived2> &n,
    const Eigen::MatrixBase<Derived3> &na)
{
    std::map<block_key, double> ret;
    assert(key.size() == 3);
    assert(n.size() == 1);
    assert(na.size() == 1);
    const int a = key(0);
    const int b = key(1);
    const int nb = key(2);
    Vector<int> v(3);
    v(0) = a;
    v(2) = n(0);
    assert(n(0) >= nb);
    for (int n1 = b; n1 <= n(0) + b - nb; ++n1)
    {
        // n1: number of derived in sample of size n
        // n2: number of ancestral "    "   "   "
        // must have: b(0) <= n1, nb(0) - b(0) <= n2 = n - n1 => n1 <= n + b - nb
        int n2 = n(0) - n1;
        v(1) = n1;
        // p(k) =  C(n_1, k) C(n_2, t - k) / C(n_1 + n_2, t)
        // gsl_ran_hypergeometric_pdf(unsigned int k, unsigned int n1, unsigned int n2, unsigned int t)
        // Here we rely on ret[bk] default constructing with value 0.
        ret[block_key(v)] += gsl_ran_hypergeometric_pdf(b, n1, n2, nb);
    }
    return ret;
}

template <size_t P>
template <typename Derived1, typename Derived2, typename Derived3>
std::map<block_key, double> marginalize_key<P>::run(
        const Eigen::MatrixBase<Derived1> &key,
        const Eigen::MatrixBase<Derived2> &n,
        const Eigen::MatrixBase<Derived3> &na)
{
    std::map<block_key, double> ret;
    assert(a(0) >= 0);
    std::map<block_key, double> sub_left = marginalize_key<1>::run(
            key.head(3),
            n.head(1),
            na.head(1));
    std::map<block_key, double> sub_right = marginalize_key<P - 1>::run(
            key.tail(3 * (P - 1)),
            n.tail(P - 1), 
            na.tail(P - 1));
    for (const auto &p_left : sub_left)
        for (const auto &p_right : sub_right)
        {
            Vector<int> v(3 * P);
            v.head(3) = p_left.first.vals;
            v.tail(3 * (P - 1)) = p_right.first.vals;
            ret[block_key(v)] += p_left.second * p_right.second;
        }
    return ret;
}

#endif
