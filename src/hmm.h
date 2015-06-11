#ifndef HMM_H
#define HMM_H

#include <cassert>
#include <cfenv>
#include <algorithm>
#include <map>
#include <unsupported/Eigen/MatrixFunctions>

#include "common.h"
#include "rate_function.h"
#include "ThreadPool.h"

typedef Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> npArray;

template <typename T>
class HMM
{
    public:
    HMM(const Vector<T> &pi, const Matrix<T> &transition, const Matrix<T> &emission,
        const int n, const int L, const int* obs, const int block_size);
    HMM(const Vector<T> &pi, const Matrix<T> &transition, const Matrix<T> &emission,
        const int n, const int L, const int* obs, const int block_size,
        const Matrix<T> &gamma, const Matrix<T> &xisum);
    T loglik(void);
    T Q(void);
    std::vector<int>& viterbi(void);
    void printobs(void);
    void computeB(void);
    void preEM(void);
    Matrix<T>& getGamma() { return gamma; }
    Matrix<T>& getXisum() { return xisum; }

    private:
    // Methods
    void forward(void);
    void backward(void);
    // Instance variables
    const Vector<T> pi;
    const Matrix<T> transition;
    const Matrix<T> emission;
    int M, n, L;
    Eigen::Map<const Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>> obs;
    const int block_size, Ltot;
    Matrix<T> B, alpha_hat, beta_hat, gamma, xisum;
    Vector<T> c;
    std::vector<int> viterbi_path;
};

template <typename T>
T compute_hmm_Q(
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
