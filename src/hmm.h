#ifndef HMM_H
#define HMM_H

#include <cassert>
#include <cfenv>
#include <algorithm>
#include <unsupported/Eigen/MatrixFunctions>

#include "common.h"
#include "conditioned_sfs.h"
#include "piecewise_exponential.h"
#include "transition.h"
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
        bool viterbi, std::vector<std::vector<int>> &viterbi_paths)
{
    // eta.print_debug();
    ThreadPool tp(numthreads);
    std::vector<HMM<T>> hmms;
    std::vector<std::thread> t;
    std::vector<std::future<T>> results;
    for (auto ob : obs)
        hmms.emplace_back(pi, transition, emission, L, ob);
    for (auto &hmm : hmms)
        results.emplace_back(tp.enqueue([&] { hmm.forward(); return hmm.loglik(); }));
    T ret = 0.0;
    for (auto &&res : results)
        ret += res.get();
    std::vector<std::future<std::vector<int>>> viterbi_results;
    if (viterbi)
    {
        for (auto &hmm : hmms)
            viterbi_results.emplace_back(tp.enqueue([&] { return hmm.viterbi(); }));
        for (auto &&res : viterbi_results)
            viterbi_paths.push_back(res.get());
    }
    return ret;
}

#endif
