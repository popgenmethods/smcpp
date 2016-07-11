#ifndef BLOCK_KEY_H
#define BLOCK_KEY_H

#include "common.h"
#include "hash.h"

struct block_key 
{ 
    block_key(const Vector<int> &vals) : vals(vals) {}
    const Vector<int> vals;

    int operator()(int k) const { return vals(k); }

    int size() const { return vals.size(); }

    friend std::ostream & operator<<(std::ostream& stream, const block_key &bk)
    {
        stream << bk.vals;
        return stream;
    }

    bool operator<(const block_key &other) const
    {
        assert(vals.size() == other.vals.size());
        for (int p = 0; p < vals.size(); ++p)
        {
            if (vals(p) != other.vals(p))
                return vals(p) < other.vals(p);
        }
        return false;
    }

    size_t hash() const
    {
        size_t h = hash_helpers::make_hash(vals(0));
        for (int p = 1; p < vals.size(); ++p)
            hash_helpers::hash_combine(h, hash_helpers::make_hash(vals(p)));
        return h;
    }

}; 

namespace std
{
    template <>
    struct hash<block_key>
    {
        size_t operator()(const block_key& bk) const
        {
            return bk.hash();
        }
    };
}

#endif
