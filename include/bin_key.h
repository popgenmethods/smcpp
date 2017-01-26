#ifndef BIN_KEY_H
#define BIN_KEY_H

#include <utility>

#include "block_key.h"

template <size_t P>
struct bin_key 
{
    template <typename Derived1>
    static block_key_prob_map run(
            const block_key &key, 
            const Eigen::MatrixBase<Derived1> &na);

    template <typename Derived1, typename Derived2>
    static block_key_prob_map run(
        const Eigen::MatrixBase<Derived1> &key, 
        const Eigen::MatrixBase<Derived2> &na);
};

template <size_t P>
template <typename Derived1>
block_key_prob_map bin_key<P>::run(
        const block_key &key, 
        const Eigen::MatrixBase<Derived1> &na)
{
    return bin_key<P>::run(key.vals, na);
}

template <>
template <typename Derived1, typename Derived2>
block_key_prob_map bin_key<1>::run(
        const Eigen::MatrixBase<Derived1> &key, 
        const Eigen::MatrixBase<Derived2> &na)
{
    Vector<int> tmp = key;
    block_key_prob_map ret;
    const int a = tmp(0);
    if (a == -1)
        for (int aa = 0; aa <= na(0); ++aa)
        {
            tmp(0) = aa;
            ret[tmp] = 1. / (double)(na(0) + 1);
        }
    else
        ret[tmp] = 1.;
    return ret;
}

template <size_t P>
template <typename Derived1, typename Derived2>
block_key_prob_map bin_key<P>::run(
        const Eigen::MatrixBase<Derived1> &key, 
        const Eigen::MatrixBase<Derived2> &na)
{
    block_key_prob_map bk1 = bin_key<1>::run(key.head(3), na.head(1));
    block_key_prob_map bk2 = bin_key<P - 1>::run(key.tail(3 * (P - 1)), na.tail(P - 1));
    block_key_prob_map ret;
    Vector<int> v(3 * P);
    for (const std::pair<block_key, double> b1 : bk1)
        for (const std::pair<block_key, double>& b2 : bk2)
        {
            v.head(3) = b1.first.vals;
            v.tail(3 * (P - 1)) = b2.first.vals;
            ret[v] += b1.second * b2.second;
        }
    return ret;
}

#endif
