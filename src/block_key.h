#ifndef BLOCK_KEY_H
#define BLOCK_KEY_H

#include <map>
#include "hash.h"

struct block_power
{
    int a, b, nb;
    bool operator==(const block_power &other) const
    {
        return std::array<int, 3>{a, b, nb} == std::array<int, 3>{other.a, other.b, other.nb};
    }
    bool operator<(const block_power &other) const
    {
        return std::array<int, 3>{a, b, nb} < std::array<int, 3>{other.a, other.b, other.nb};
    }
    friend std::ostream& operator<< (std::ostream& stream, const block_power& bp) {
        stream << "{" << bp.a << "," << bp.b << "/" << bp.nb << "}";
        return stream;
    }
};

struct block_key
{
    bool alt_block;
    std::map<block_power, int> powers;
    bool operator==(const block_key &other) const 
    { 
        return alt_block == other.alt_block and powers == other.powers;
    }
    friend std::ostream& operator<< (std::ostream& stream, const block_key& bk) {
        stream << "(" << bk.alt_block << ", " << bk.powers << ")";
        return stream;
    }
};

namespace std
{
    template <>
    struct hash<block_power>
    {
        size_t operator()(const block_power& bp) const
        {
            size_t h = hash_helpers::make_hash(bp.a);
            hash_helpers::hash_combine(h, hash_helpers::make_hash(bp.b));
            hash_helpers::hash_combine(h, hash_helpers::make_hash(bp.nb));
            return h;
        }
    };
    template <>
    struct hash<block_key>
    {
        size_t operator()(const block_key& bk) const
        {
            size_t h = hash_helpers::make_hash(bk.alt_block);
            hash_helpers::hash_combine(h, hash_helpers::make_hash(bk.powers));
            return h;
        }
    };
}

typedef std::vector<std::pair<bool, std::vector<std::pair<block_power, int> > > > block_key_vector;

#endif
