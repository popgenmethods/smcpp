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
        const int mask_freq);
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
    const int block_size;
    const Vector<adouble> *pi;
    const Matrix<adouble> *transition, *emission;
    const Matrix<int> emission_mask, two_mask;
    const int mask_freq, M, Ltot;
    std::vector<Vector<adouble>*> Bptr;
    Matrix<adouble> B;
    Matrix<float> alpha_hat, xisum;
    Vector<float> c;
    Vector<float> gamma0;

    std::unordered_map<block_key, Vector<adouble> > block_prob_map;
    std::vector<std::pair<bool, decltype(block_key::powers)> > block_keys;
    std::unordered_map<block_key, unsigned long> comb_coeffs;
    std::map<Vector<adouble>*, Vector<float> > gamma_sums;
    friend class InferenceManager;
};

#endif
