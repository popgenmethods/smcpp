#include "hmm.h"
#include <csignal>
#include <gperftools/profiler.h>

#define PRINT_VECTOR(v) std::cout << v.template cast<double>().transpose() << std::endl;
void print_matrix(Matrix<adouble> &M) { std::cout << M.cast<double>() << std::endl << std::endl; }

class CondensedObsIter
{
    private:
    class iter 
    {
        public:
        iter(int n, const Eigen::Matrix<int, Eigen::Dynamic, 3> &obs,
            int pos, int lt) : _n(n), _obs(obs), _pos(pos), _lt(lt) {}

        bool operator!= (const iter& other) const
        {
            return _pos != other._pos or _lt != other._lt;
        }

        const int operator*() const
        {
            return emission_index(_n, _obs(_pos, 1), _obs(_pos, 2));
        }

        const iter& operator++ ()
        {
            if (_lt == _obs(_pos, 0) - 1)
            {
                _lt = 0;
                _pos++;
            }
            else
                _lt++;
            return *this;
        }

        private:
        int _n;
        const Eigen::Matrix<int, Eigen::Dynamic, 3> _obs;
        int _pos, _lt;
    };
    const int _n;
    const Eigen::Matrix<int, Eigen::Dynamic, 3> _obs;
    public:
    CondensedObsIter(const int n, const Eigen::Matrix<int, Eigen::Dynamic, 3> &obs) : _n(n), _obs(obs) {}
    iter begin() const { return iter(_n, _obs, 0, 0); }
    iter end() const { return iter(_n, _obs, _obs.rows(), 0); }
};

template <typename T>
HMM<T>::HMM(const Vector<T> &pi, const Matrix<T> &transition, const Matrix<T> &emission,
        const int n, const int L, const int* obs, const int block_size) :
    pi(pi), transition(transition), emission(emission), M(emission.rows()),
    n(n), L(L), obs(obs, L, 3), block_size(block_size), Ltot(ceil(this->obs.col(0).sum() / block_size)), 
    B(M, Ltot), alpha_hat(M, Ltot), beta_hat(M, Ltot), c(Ltot) { }

    template <typename T>
T HMM<T>::loglik()
{
    forward();
    return c.array().log().sum();
}

    template <typename T>
std::vector<int>& HMM<T>::viterbi(void)
{
    // Compute logs of transition and emission matrices 
    // Do not bother to record derivatives since we don't use them for Viterbi algorithm
    feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);
    Eigen::MatrixXd log_transition = transition.template cast<double>().array().log();
    Eigen::MatrixXd log_emission = emission.template cast<double>().array().log();
    std::vector<double> V(M), V1(M);
    std::vector<std::vector<int>> path(M), newpath(M);
    Eigen::MatrixXd pid = pi.template cast<double>();
    std::vector<int> zeros(M, 0);
    double p, p2, lemit;
    int st;
    for (int m = 0; m < M; ++m)
    {
        V[m] = log(pid(m)) + log_emission(m, emission_index(n, obs(0,1), obs(0,2)));
        path[m] = zeros;
        path[m][m]++;
    }
    for (int ell = 0; ell < L; ++ell)
    {
        int R = obs(ell, 0);
        if (ell == 0)
            R--;
        for (int r = 0; r < R; ++r)
        {
            for (int m = 0; m < M; ++m)
            {
                lemit = log_emission(m, emission_index(n, obs(ell, 1), obs(ell, 2)));
                p = -INFINITY;
                st = 0;
                for (int m2 = 0; m2 < M; ++m2)
                {
                    p2 = V[m2] + log_transition(m2, m) + lemit;
                    if (p2 > p)
                    {
                        p = p2;
                        st = m2;
                    }
                }
                V1[m] = p;
                newpath[m] = path[st];
                newpath[m][m]++;
            }
            path = newpath;
            V = V1;
        }
    }
    p = -INFINITY;
    st = 0;
    for (int m = 0; m < M; ++m)
    {
        if (V[m] > p)
        {
            p = V[m];
            st = m;
        }
    }
    viterbi_path = path[st];
    return viterbi_path;
}

    template <typename T>
void HMM<T>::forward(void)
{
    std::cout << "forward algorithm... " << std::endl;
    //ProfilerStart("Bprof");
    c = Vector<T>::Zero(Ltot);
    B = Matrix<T>::Ones(M, Ltot);
    // Create obs vectors based on blocking
    // CondensedObsIter citer(n, obs);
    int block = 0;
    int i = 0, ob, R;
    std::map<int, int> powers;
    for (int ell = 0; ell < L; ++ell)
    {
        R = obs(ell, 0);
        ob = emission_index(n, obs(ell, 1), obs(ell, 2));
        for (int r = 0; r < R; ++r)
        {
            powers[ob]++;
            if (++i == block_size)
            {
                i = 0;
                for (auto &p : powers)
                    B.col(block) = B.col(block).cwiseProduct(
                            emission.col(p.first).array().pow(p.second).matrix());
                powers.clear();
                block++;
            }
        }
    }
    //ProfilerStop();
    // Run forward algorithm
    alpha_hat.col(0) = B.col(0).asDiagonal() * transition.transpose() * pi;
	c(0) = alpha_hat.col(0).sum();
    alpha_hat.col(0) /= c(0);
    for (int ell = 1; ell < Ltot; ++ell)
    {
        alpha_hat.col(ell) = B.col(ell).asDiagonal() * transition.transpose() * alpha_hat.col(ell - 1);
        c(ell) = alpha_hat.col(ell).sum();
        if (isnan(toDouble(c(ell))))
            throw std::domain_error("something went wrong in forward algorithm");
        alpha_hat.col(ell) /= c(ell);
    }
}

template <typename T>
void HMM<T>::backward(void)
{
    std::cout << "backward algorithm... " << std::endl;
    beta_hat.col(Ltot - 1) = Vector<T>::Ones(M);
    for (int ell = Ltot - 2; ell >= 0; --ell)
        beta_hat.col(ell) = transition * B.col(ell + 1).asDiagonal() * beta_hat.col(ell + 1) / c(ell + 1);
}

template <typename T>
T HMM<T>::Q(void)
{
    std::cout << "Q..." << std::endl;
	Matrix<T> log_transition = transition.array().log();
	Matrix<T> gamma = alpha_hat.cwiseProduct(beta_hat);
    Matrix<T> tmp, xisum = Matrix<T>::Zero(M, M);
    // Begin Baum-Welch algorithm
	T ret = gamma.col(0).dot(pi.array().log().matrix());
    for (int ell = 0; ell < Ltot; ++ell)
    {
        ret += B.col(ell).array().log().matrix().dot(gamma.col(ell));
        if (ell > 0)
            xisum += c(ell) * alpha_hat.col(ell - 1) * 
                beta_hat.col(ell).transpose() * B.col(ell).asDiagonal();
    }
    if (isnan(toDouble(ret)))
        throw std::domain_error("something went wrong in Q");
    ret += log_transition.cwiseProduct(xisum).sum();
    return ret;
}

template <typename T>
T compute_hmm_Q(
        const Vector<T> &pi, const Matrix<T> &transition, const Matrix<T> &emission, 
        const int n, const int L, const std::vector<int*> obs,
        int block_size,
        int numthreads, 
        std::vector<Vector<double>> &cs,
        std::vector<Matrix<double>> &alpha_hats, 
        std::vector<Matrix<double>> &beta_hats,
        std::vector<Matrix<double>> &Bs,
        bool compute_alpha_beta)
{
    // eta.print_debug();
    ThreadPool tp(numthreads);
    std::vector<HMM<T>> hmms;
    std::vector<std::thread> t;
    std::vector<std::future<T>> results;
    std::vector<T> results_unthreaded;
    for (int i = 0; i < obs.size(); ++i)
    {
        hmms.emplace_back(pi, transition, emission, n, L, obs[i], block_size);
        if (not compute_alpha_beta)
        {
            hmms.back().c = cs[i].template cast<T>();
            hmms.back().alpha_hat = alpha_hats[i].template cast<T>();
            hmms.back().beta_hat = beta_hats[i].template cast<T>();
            hmms.back().B = Bs[i].template cast<T>();
        }
    }
    for (auto &hmm : hmms)
        if (numthreads == 0)
        {
            if (compute_alpha_beta)
            {
                hmm.forward(); 
                hmm.backward();
            }
            results_unthreaded.push_back(hmm.Q());
        }
        else
        {
            results.emplace_back(tp.enqueue([&] { 
                if (compute_alpha_beta)
                {
                    hmm.forward(); 
                    hmm.backward();
                }
                return hmm.Q(); 
            }));
        }
    T ret = 0.0;
    if (numthreads == 0)
        for (auto &&res : results_unthreaded)
            ret += res;
    else
        for (auto &&res : results)
            ret += res.get();
    if (compute_alpha_beta)
    {
        cs.clear();
        alpha_hats.clear();
        beta_hats.clear();
        Bs.clear();
        for (int i = 0; i < hmms.size(); ++i)
        {
            auto hmm = hmms[i];
            cs.push_back(hmm.c.template cast<double>());
            alpha_hats.push_back(hmm.alpha_hat.template cast<double>());
            beta_hats.push_back(hmm.beta_hat.template cast<double>());
            Bs.push_back(hmm.B.template cast<double>());
        }
    }
    return ret;
}

template double compute_hmm_Q(
        const Vector<double> &pi, const Matrix<double> &transition, const Matrix<double> &emission, 
        const int n, const int L, const std::vector<int*> obs,
        int block_size,
        int numthreads, 
        std::vector<Vector<double>> &cs, 
        std::vector<Matrix<double>> &alpha_hats, 
        std::vector<Matrix<double>> &beta_hats,
        std::vector<Matrix<double>> &Bs,
        bool compute_alpha_beta);

template adouble compute_hmm_Q(
        const Vector<adouble> &pi, const Matrix<adouble> &transition, const Matrix<adouble> &emission, 
        const int n, const int L, const std::vector<int*> obs,
        int block_size,
        int numthreads, 
        std::vector<Vector<double>> &cs, 
        std::vector<Matrix<double>> &alpha_hats, 
        std::vector<Matrix<double>> &beta_hats,
        std::vector<Matrix<double>> &Bs,
        bool compute_alpha_beta);

template <typename T>
T compute_hmm_likelihood(
        const Vector<T> &pi, const Matrix<T> &transition, const Matrix<T> &emission, 
        const int n, const int L, const std::vector<int*> obs,
        int block_size,
        int numthreads, 
        bool viterbi, std::vector<std::vector<int>> &viterbi_paths)
{
    // eta.print_debug();
    ThreadPool tp(numthreads);
    std::vector<HMM<T>> hmms;
    std::vector<std::thread> t;
    std::vector<std::future<T>> results;
    for (auto ob : obs)
        hmms.emplace_back(pi, transition, emission, n, L, ob, block_size);
    for (auto &hmm : hmms)
        results.emplace_back(tp.enqueue([&] { return hmm.loglik(); }));
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

template double compute_hmm_likelihood(
        const Vector<double> &pi, const Matrix<double> &transition,
        const Matrix<double> &emission, 
        const int n, const int L, const std::vector<int*> obs,
        const int block_size,
        int numthreads, 
        bool viterbi, std::vector<std::vector<int>> &viterbi_paths);

template adouble compute_hmm_likelihood(
        const Vector<adouble> &pi, const Matrix<adouble> &transition, const Matrix<adouble> &emission, 
        const int n, const int L, const std::vector<int*> obs,
        const int block_size,
        int numthreads, 
        bool viterbi, std::vector<std::vector<int>> &viterbi_paths);

template class HMM<double>;
template class HMM<adouble>;


/*
int main(int argc, char** argv)
{
    Eigen::Matrix2d transition;
    transition << 0.7, 0.3,
                  0.3, 0.7;
    Eigen::Vector2d pi;
    pi << 0.5, 0.5;
    int L = 5; 
    int obs[5][3] = { {1, 0, 0}, {1, 0, 0}, {1, 0, 1}, {1, 0, 0}, {1, 0, 0} };
    std::vector<Matrix<double>> emission(2, Matrix<double>::Zero(1, 2));
    emission[0] << 0.9, 0.1;
    emission[1] << 0.2, 0.8;
    HMM<double> hmm(pi, transition, emission, L, obs[0]);
    hmm.forward();
    hmm.backward();
    std::cout << hmm.Q() << std::endl << std::endl;
    std::cout << hmm.alpha_hat << std::endl << std::endl;
    std::cout << hmm.beta_hat << std::endl << std::endl;
    std::cout << hmm.alpha_hat.cwiseProduct(hmm.beta_hat) << std::endl << std::endl;
    std::cout << hmm.alpha_hat.cwiseProduct(hmm.beta_hat).colwise().sum() << std::endl << std::endl;
    std::cout << hmm.c.array().log().sum() << std::endl << std::endl;
    hmm.fast_forward();
    std::cout << hmm.c.array().log().sum() << std::endl << std::endl;
}
*/

