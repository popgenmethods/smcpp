#ifndef HMM_H
#define HMM_H

#include <unordered_map>
#include <map>
#include "common.h"

// Allow for hashing of map<T, U>
// a little helper that should IMHO be standardized
namespace
{
    template<typename T>
    std::size_t make_hash(const T& v)
    {
        return std::hash<T>()(v);
    }

    // adapted from boost::hash_combine
    void hash_combine(std::size_t& h, const std::size_t& v)
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
            size_t h=make_hash(v.first);
            hash_combine(h, make_hash(v.second));
            return h;
        }
    };
    template<typename... T>
    struct hash<map<T...>> : hash_container<map<T...>> {};
}


class InferenceManager;

class HMM
{
    public:
    HMM(Eigen::Matrix<int, Eigen::Dynamic, 2> obs, const int block_size,
        const Vector<adouble> *pi, const Matrix<adouble> *transition, 
        const Matrix<adouble> *emission, const Matrix<adouble> *emission_mask, 
        const int mask_freq, const int mask_offset);
    void Estep(void);
    double loglik(void);
    adouble Q(void);
    // std::vector<int>& viterbi(void);

    private:
    HMM(HMM const&) = delete;
    HMM& operator=(HMM const&) = delete;
    // Methods
    void prepare_B(void);
    void recompute_B(void);
    void forward_backward(void);
    void domain_error(double);

    Eigen::Matrix<int, Eigen::Dynamic, 2> obs;
    const int block_size;

    // Instance variables
    const Vector<adouble> *pi;
    const Matrix<adouble> *transition, *emission, *emission_mask;
    const int mask_freq, mask_offset, M, Ltot;
    std::vector<Vector<adouble>*> Bptr;
    std::vector<Eigen::Array<adouble, Eigen::Dynamic, 1>*> logBptr;
    Matrix<adouble> B;
    Matrix<double> alpha_hat, beta_hat, gamma, xisum;
    Vector<double> c;
    std::vector<int> viterbi_path;
    std::unordered_map<std::pair<bool, std::map<int, int> >, std::pair<Vector<adouble>, Eigen::Array<adouble, Eigen::Dynamic, 1> > > block_prob_map;
    std::unordered_map<Eigen::Array<adouble, Eigen::Dynamic, 1>*, decltype(block_prob_map)::key_type> reverse_map;
    std::unordered_map<std::pair<bool, std::map<int, int> >, int > block_prob_counts;
    friend class InferenceManager;
};

template <typename T>
Eigen::Matrix<T, 2, 1> compute_hmm_Q(
        const Vector<T> &pi, const Matrix<T> &transition,
        const Matrix<T> &emission,
        const int n, const int L, const std::vector<int*> obs,
        int block_size,
        int numthreads, 
        std::vector<Matrix<double>> &gammas,
        std::vector<Matrix<double>> &xisums,
        bool recompute);

/*
template <typename T>
T compute_hmm_likelihood(
        const Vector<T> &pi, const Matrix<T> &transition,
        const std::vector<Matrix<T>>& emission, 
        const int L, const std::vector<int*> obs,
        int block_size,
        int numthreads, 
        bool viterbi, std::vector<std::vector<int>> &viterbi_paths);
        */

#endif
