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
    T loglik(void);
    T Q(void);
    T Q(const Matrix<T> &, const Matrix<T> &);
    std::vector<int>& viterbi(void);
    void printobs(void);
    void fast_forward(void);
    void forward(void);
    void backward(void);

    private:
    // Methods
    Matrix<T> matpow(const Matrix<T>&, int);
    void diag_obs(int);
    Matrix<T> O0Tpow(int);

    // Instance variables
    private:
    std::map<std::pair<int, int>, Eigen::DiagonalMatrix<T, Eigen::Dynamic>> obs_cache;
    const Vector<T> pi;
    const Matrix<T> transition;
    const Matrix<T> emission;
    int M, n, L;
    Eigen::Map<const Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>> obs;
    const int block_size, Ltot;

    public:
    Matrix<T> B, alpha_hat, beta_hat;
    Vector<T> c;

    private:
    std::vector<int> viterbi_path;
    std::map<int, Matrix<T>> O0Tpow_memo;

    friend int main(int, char**);
};

template <typename T>
T compute_hmm_Q(
        const Vector<T> &pi, const Matrix<T> &transition,
        const Matrix<T> &emission,
        const int n, const int L, const std::vector<int*> obs,
        int block_size,
        int numthreads, 
        std::vector<Vector<double>> &cs,
        std::vector<Matrix<double>> &alpha_hats, 
        std::vector<Matrix<double>> &beta_hats,
        std::vector<Matrix<double>> &Bs,
        bool compute_alpha_beta);

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
