#ifndef HMM_H
#define HMM_H

#include <unordered_map>
#include <map>
#include <gmpxx.h>

#include "common.h"
#include "hash.h"

struct block_key
{
    bool alt_block;
    std::map<std::pair<int, int>, int> powers;
    bool operator==(const block_key &other) const 
    { 
        return alt_block == other.alt_block and powers == other.powers;
    }
};

namespace std
{
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

class InferenceManager;

class HMM
{
    public:
    HMM(const Matrix<int> &obs, const int n, const int block_size,
        const Vector<adouble> *pi, const Matrix<adouble> *transition, 
        const Matrix<adouble> *emission, const Matrix<int> emission_mask, 
        const int mask_freq, const int mask_offset);
    void Estep(void);
    double loglik(void);
    adouble Q(void);
    // std::vector<int>& viterbi(void);
    void fill_B(void) { B = Matrix<adouble>::Zero(M, Ltot); for (int ell = 0; ell < Ltot; ++ell) B.col(ell) = *Bptr[ell]; }

    private:
    HMM(HMM const&) = delete;
    HMM& operator=(HMM const&) = delete;
    // Methods
    void prepare_B(const Matrix<int>&);
    void recompute_B(void);
    void forward_backward(void);
    void domain_error(double);
    bool is_alt_block(int);

    // Instance variables
    const int n;
    const int block_size, alt_block_size;
    const Vector<adouble> *pi;
    const Matrix<adouble> *transition, *emission;
    const Matrix<int> emission_mask, two_mask;
    const int mask_freq, mask_offset, M, Ltot;
    std::vector<Vector<adouble>*> Bptr;
    std::vector<Eigen::Array<adouble, Eigen::Dynamic, 1>*> logBptr;
    std::vector<Vector<double>*> dBptr;
    Matrix<adouble> B;
    Matrix<double> alpha_hat, beta_hat, gamma, xisum, xisum_alt;
    Vector<double> c;
    std::vector<int> viterbi_path;
    std::unordered_map<block_key, std::tuple<Vector<adouble>, Eigen::Array<adouble, Eigen::Dynamic, 1>, Vector<double> > > block_prob_map;
    std::vector<block_key> block_prob_map_keys;
    std::vector<std::pair<bool, decltype(block_key::powers)> > block_keys;
    std::unordered_map<block_key, unsigned long> comb_coeffs;
    // std::unordered_map<Eigen::Array<adouble, Eigen::Dynamic, 1>*, decltype(block_prob_map)::key_type> reverse_map;
    std::vector<std::pair<Eigen::Array<adouble, Eigen::Dynamic, 1>*, std::vector<int> > > block_pairs;
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
