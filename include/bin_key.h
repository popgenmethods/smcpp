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
            const Eigen::MatrixBase<Derived1> &na,
            const double cutoff);

    template <typename Derived1, typename Derived2>
    static std::set<block_key> run(
        const Eigen::MatrixBase<Derived1> &key, 
        const Eigen::MatrixBase<Derived2> &na,
        const double cutoff);
};

template <size_t P>
template <typename Derived1>
std::set<block_key> bin_key<P>::run(
        const block_key &key, 
        const Eigen::MatrixBase<Derived1> &na,
        const double cutoff)
{
    return bin_key<P>::run(key.vals, na, cutoff);
}

template <>
template <typename Derived1, typename Derived2>
std::set<block_key> bin_key<1>::run(
        const Eigen::MatrixBase<Derived1> &key, 
        const Eigen::MatrixBase<Derived2> &na,
        const double cutoff)
{
    Vector<int> tmp = key;
    std::set<block_key> ret;
    const int a = tmp(0);
    const int b = tmp(1);
    const int nb = tmp(2);
    if (a == -1)
        for (int aa = 0; aa <= na(0); ++aa)
        {
            tmp(0) = aa;
            std::set<block_key> s = bin_key<1>::run(tmp, na, cutoff);
            ret.insert(s.begin(), s.end());
        }
    else
    {
        ret.emplace(key);
        if (true) // or undistinguish)
            for (int aa = 0; aa < 2; ++aa)
            {
                int bb = key(1) - aa;
                tmp(0) = aa;
                if ((aa + bb == key(0) + key(1)) and
                        (0 <= bb) and
                        (bb <= key(2)))
                    ret.emplace(tmp);
            }
        /*
        if (nb > 0 and ((double)b / (double)nb > cutoff))
            for (int bb = (int)(cutoff * nb); bb <= nb; ++bb)
            {
                tmp(1) = bb;
                ret.emplace(tmp);
            }
        */
    }
    return ret;
}

template <size_t P>
template <typename Derived1, typename Derived2>
std::set<block_key> bin_key<P>::run(
        const Eigen::MatrixBase<Derived1> &key, 
        const Eigen::MatrixBase<Derived2> &na,
        const double cutoff)
{
    std::set<block_key> bk1 = bin_key<1>::run(key.head(3), na.head(1), cutoff);
    std::set<block_key> bk2 = bin_key<P - 1>::run(key.tail(3 * (P - 1)), na.tail(P - 1), cutoff);
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
