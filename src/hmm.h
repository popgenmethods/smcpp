#ifndef HMM_H
#define HMM_H

#include <unordered_map>
#include <map>

#include "common.h"
#include "inference_manager.h"
#include "block_key.h"

class InferenceManager;

class HMM
{
    public:
    HMM(const Matrix<int> &obs, const int n, const int block_size,
        const Vector<adouble> *pi, const Matrix<adouble> *transition, 
        const int mask_freq, InferenceManager* im);
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
    void forward_only(void);
    void forward_backward(void);
    void domain_error(double);
    bool is_alt_block(int);

    // Instance variables
    const int n, block_size, alt_block_size;
    const Vector<adouble> *pi;
    const Matrix<adouble> *transition;
    const int mask_freq, M, Ltot;
    std::vector<Vector<adouble>*> Bptr;
    std::map<Vector<adouble>*, block_key> bmap;
    Matrix<adouble> B;
    Matrix<fbType> alpha_hat, xisum, xisum_alt, gamma;
    Vector<fbType> gamma0;
    Vector<double> c;
    InferenceManager* im;

    block_key_vector block_keys;
    std::map<Vector<adouble>*, Vector<fbType> > gamma_sums;
    friend class InferenceManager;
};

#endif
