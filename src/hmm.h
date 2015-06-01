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
    HMM(const Vector<T> &pi, const Matrix<T> &transition,
        const std::vector<Matrix<T>> &emission,
        const int L, const int* obs);
    T loglik(void);
    T Q(void);
    std::vector<int>& viterbi(void);
    void printobs(void);

    private:
    // Methods
    Matrix<T> matpow(const Matrix<T>&, int);
    void diag_obs(int);
    Matrix<T> O0Tpow(int);
    void fast_forward(void);
    void forward(void);
    void backward(void);

    // Instance variables
    private:
    const Vector<T> pi;
    const Matrix<T> transition;
    const std::vector<Matrix<T>> emission;
    int M, L;
    Eigen::Map<const Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>> obs;
    int Ltot;
    Eigen::DiagonalMatrix<T, Eigen::Dynamic> D;
    Matrix<T> O0T, alpha_hat, beta_hat;
    Vector<T> c;
    std::vector<int> viterbi_path;
    std::map<int, Matrix<T>> O0Tpow_memo;

    friend int main(int, char**);
};

template <typename T>
T compute_hmm_likelihood(
        const Vector<T> &pi, const Matrix<T> &transition,
        const std::vector<Matrix<T>>& emission, 
        const int L, const std::vector<int*> obs,
        int numthreads, 
        bool viterbi, std::vector<std::vector<int>> &viterbi_paths);

#endif
