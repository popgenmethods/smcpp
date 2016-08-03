#ifndef HASH_H
#define HASH_H

#include <map>

namespace hash_helpers
{
    template<typename T>
    std::size_t make_hash(const T& v)
    {
        return std::hash<T>()(v);
    }

    // adapted from boost::hash_combine
    inline void hash_combine(std::size_t& h, const std::size_t& v)
    {
        h ^= v + 0x9e3779b9 + (h << 6) + (h >> 2);
    }

    // hash any container
    template<typename T>
    struct hash_container
    {
        size_t operator()(const T& v) const
        {
            size_t h=0;
            for( const auto& e : v ) {
                hash_combine(h, make_hash(e));
            }
            return h;
        }
    };
}

// the same for map<T,U> if T and U are hashable
namespace std
{
    template<typename T, typename U>
    struct hash<pair<T, U>>
    {
        size_t operator()(const pair<T,U>& v) const
        {
            size_t h = hash_helpers::make_hash(v.first);
            hash_helpers::hash_combine(h, hash_helpers::make_hash(v.second));
            return h;
        }
    };
    template<typename... T>
    struct hash<map<T...>> : hash_helpers::hash_container<map<T...>> {};
}

#endif
