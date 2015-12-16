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
        const Matrix<adouble> *emission,
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
    const int n;
    const int block_size;
    const Vector<adouble> *pi;
    const Matrix<adouble> *transition, *emission;
    const int mask_freq, M, Ltot;
    std::vector<Vector<adouble>*> Bptr;
    Matrix<adouble> B;
    Matrix<float> alpha_hat, xisum, gamma;
    Vector<float> c, gamma0;
    InferenceManager* im;

    std::vector<std::pair<bool, decltype(block_key::powers)> > block_keys;
    std::map<Vector<adouble>*, Vector<float> > gamma_sums;
    friend class InferenceManager;
};

#endif
