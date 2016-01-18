#ifndef HMM_H
#define HMM_H

#include <unordered_map>
#include <map>

#include "common.h"
#include "inference_manager.h"
#include "inference_bundle.h"
#include "transition_bundle.h"

class InferenceManager;

class HMM
{
    public:
    HMM(const int hmm_num, const Matrix<int> &obs, const InferenceBundle *ib);
    void Estep(void);
    double loglik(void);
    adouble Q(void);

    private:
    HMM(HMM const&) = delete;
    HMM& operator=(HMM const&) = delete;
    // Methods
    void forward_backward(void);
    void domain_error(double);
    inline block_key ob_key(int i) { return {obs(i, 1), obs(i, 2), obs(i, 3)}; }

    // Instance variables
    const int hmm_num;
    const Matrix<int> obs;
    const InferenceBundle *ib;
    const int M, L;
    Matrix<double> alpha_hat, xisum, gamma;
    Vector<double> c, gamma0;
    std::map<block_key, Vector<double> > gamma_sums;
    friend class InferenceManager;
};

#endif
