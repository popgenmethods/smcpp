#ifndef BIN_KEY_H
#define BIN_KEY_H

#include <utility>

#include "block_key.h"

template <size_t P>
struct bin_key 
{
    template <typename Derived1>
    static std::set<block_key> run(
            const block_key &key, 
            const Eigen::MatrixBase<Derived1> &na);

    template <typename Derived1, typename Derived2>
    static std::set<block_key> run(
        const Eigen::MatrixBase<Derived1> &key, 
        const Eigen::MatrixBase<Derived2> &na);
};

template <size_t P>
template <typename Derived1>
std::set<block_key> bin_key<P>::run(
        const block_key &key, 
        const Eigen::MatrixBase<Derived1> &na)
{
    return bin_key<P>::run(key.vals, na);
}

template <>
template <typename Derived1, typename Derived2>
std::set<block_key> bin_key<1>::run(
        const Eigen::MatrixBase<Derived1> &key, 
        const Eigen::MatrixBase<Derived2> &na)
{
    Vector<int> tmp = key;
    std::set<block_key> ret;
    const int a = tmp(0);
    if (a == -1)
        for (int aa = 0; aa <= na(0); ++aa)
        {
            tmp(0) = aa;
            ret.emplace(tmp);
        }
    else
        ret.emplace(key);
    return ret;
}

template <size_t P>
template <typename Derived1, typename Derived2>
std::set<block_key> bin_key<P>::run(
        const Eigen::MatrixBase<Derived1> &key, 
        const Eigen::MatrixBase<Derived2> &na)
{
    std::set<block_key> bk1 = bin_key<1>::run(key.head(3), na.head(1));
    std::set<block_key> bk2 = bin_key<P - 1>::run(key.tail(3 * (P - 1)), na.tail(P - 1));
    std::set<block_key> ret;
    Vector<int> v(3 * P);
    for (const block_key b1 : bk1)
        for (const block_key b2 : bk2)
        {
            v.head(3) = b1.vals;
            v.tail(3 * (P - 1)) = b2.vals;
            ret.emplace(v);
        }
    return ret;
}

#endif
