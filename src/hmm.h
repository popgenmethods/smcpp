#ifndef HMM_H
#define HMM_H

#include <cassert>
#include <cfenv>
#include <algorithm>
#include <map>
#include <unsupported/Eigen/MatrixFunctions>

#include "common.h"
#include "piecewise_exponential.h"
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
    std::vector<int>& viterbi(void);
    void forward(void);
    void printobs(void);

    private:
    // Methods
    Matrix<T> matpow(const Matrix<T>&, int);
    template <int s>
    void diag_obs(Eigen::DiagonalMatrix<T, s> &D, int a, int b);
    Matrix<T> O0Tpow(int);

    // Instance variables
    const Vector<T> pi;
    const Matrix<T> transition;
    const std::vector<Matrix<T>> emission;
    int M, L;
    Eigen::Map<const Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>> obs;
    std::vector<T> logc;
    std::vector<int> viterbi_path;
    Matrix<T> O0T;
    std::map<int, Matrix<T>> O0Tpow_memo;
};

template <typename T>
T compute_hmm_likelihood(
        const PiecewiseExponential<T> &eta, 
        const Vector<T> &pi, const Matrix<T> &transition,
        const std::vector<Matrix<T>>& emission, 
        const int L, const std::vector<int*> obs,
        int numthreads, 
        bool viterbi, std::vector<std::vector<int>> &viterbi_paths);

#endif
