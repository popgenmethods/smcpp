#ifndef HMM_H
#define HMM_H

#include <map>
#include "common.h"

typedef Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> npArray;

class InferenceManager;

class HMM
{
    public:
    HMM(Eigen::Matrix<int, Eigen::Dynamic, 2> obs, const int block_size,
        const Vector<adouble> *pi, const Matrix<adouble> *transition, 
        const Matrix<adouble> *emission);
    void Estep(void);
    double loglik(void);
    adouble Q(void);
    // std::vector<int>& viterbi(void);

    private:
    // Methods
    void recompute_B(void);
    void forward_backward(void);

    Eigen::Matrix<int, Eigen::Dynamic, 2> obs;
    const int block_size;

    // Instance variables
    const Vector<adouble> *pi;
    const Matrix<adouble> *transition, *emission;
    const int M, Ltot;
    Matrix<adouble> B;
    Matrix<double> alpha_hat, beta_hat, gamma, xisum;
    Vector<double> c;
    std::vector<int> viterbi_path;

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
