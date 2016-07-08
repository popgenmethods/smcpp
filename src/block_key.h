#ifndef BLOCK_KEY_H
#define BLOCK_KEY_H

#include <Eigen/Dense>
#include "hash.h"

template <size_t P>
using block_key = Vector<int, 1 + 2 * P>;

namespace std
{
    template <size_t P>
    struct hash<block_key<P> >
    {
        size_t operator()(const block_key<P>& bk) const
        {
            size_t h = hash_helpers::make_hash(bk(0));
            for (int p = 1; p < 1 + 2 * P; ++p)
                hash_helpers::hash_combine(h, hash_helpers::make_hash(bk(p));
            return h;
        }
    };
}

#endif
