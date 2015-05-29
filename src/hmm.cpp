#include "hmm.h"
#include <csignal>

template <typename T>
HMM<T>::HMM(const Vector<T> &pi, const Matrix<T> &transition, 
        const std::vector<Matrix<T>> &emission,
        const int L, const int* obs) :
    pi(pi), transition(transition), emission(emission), M(emission.size()),
    L(L), obs(obs, L, 3), Ltot(this->obs.col(0).sum()), D(M), O0T(M, M), 
	alpha_hat(M, Ltot), beta_hat(M, Ltot), c(L)
{ 
    feraiseexcept(FE_ALL_EXCEPT & ~FE_INEXACT);
    D.setZero();
    for (int m = 0; m < M; ++m)
    {
        D.diagonal()(m) = emission[m](0, 0);
        assert(emission[m](a,b)>=0);
        assert(emission[m](a,b)<=1);
    }
    O0T = D * transition.transpose();
	alpha_hat.fill(0.0);
	beta_hat.fill(0.0);
}


template <typename T>
T HMM<T>::loglik()
{
	fast_forward();
	return c.array().log().sum();
}

template <typename T>
std::vector<int>& HMM<T>::viterbi(void)
{
    // Compute logs of transition and emission matrices 
    // Do not bother to record derivatives since we don't use them for Viterbi algorithm
    std::vector<Eigen::MatrixXd> log_emission(L);
    Eigen::MatrixXd log_transition = transition.template cast<double>().array().log();
    std::transform(emission.begin(), emission.end(), log_emission.begin(), 
            [](decltype(emission[0]) &x) -> Eigen::MatrixXd {return x.template cast<double>().array().log();});
    std::vector<double> V(M), V1(M);
    std::vector<std::vector<int>> path(M), newpath(M);
    Eigen::MatrixXd pid = pi.template cast<double>();
    std::vector<int> zeros(M, 0);
    double p, p2, lemit;
    int st;
    for (int m = 0; m < M; ++m)
    {
        V[m] = log(pid(m)) + log_emission[m](obs(0,1), obs(0,2));
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
                lemit = log_emission[m](obs(ell, 1), obs(ell, 2));
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
Matrix<T> HMM<T>::O0Tpow(int p)
{
    Matrix<T> A;
    if (p == 0)
        return Matrix<T>::Identity(M, M);
    if (p == 1)
        return O0T;
    if (O0Tpow_memo.count(p) == 0)
    {
        if (p % 2 == 0)
        {
            A = O0Tpow(p / 2);
            O0Tpow_memo[p] = A * A;
        }
        else
        {
            A = O0Tpow(p - 1);
            O0Tpow_memo[p] = O0T * A;
        }
    }
    return O0Tpow_memo[p];
}

template <typename T>
Matrix<T> HMM<T>::matpow(const Matrix<T> &A, int pp)
{
    Matrix<T> m_tmp = A;
    Matrix<T> res = Matrix<T>::Identity(M, M);
    while (pp >= 1) {
        if (std::fmod(pp, 2) >= 1)
            res = m_tmp * res;
        m_tmp *= m_tmp;
        pp /= 2;
    }
    return res;
}

template <typename T>
void HMM<T>::diag_obs(int ell)
{
	int a = obs(ell, 1);
	int b = obs(ell, 2);
    for (int m = 0; m < M; ++m)
    {
        D.diagonal()(m) = emission[m](a, b);
        assert(emission[m](a,b)>=0);
        assert(emission[m](a,b)<=1);
    }
}

template <typename T>
void HMM<T>::fast_forward(void)
{
    D.setZero();
    int p;
    diag_obs(0);
    Vector<T> ahat = D * pi;
	Matrix<T> P;
    for (int ell = 0; ell < L; ++ell)
    {
        p = obs(ell, 0);
		if (ell == 0)
			p--;
        if (obs(ell, 1) == 0 && obs(ell, 2) == 0)
			P = O0Tpow(p);
        else    
        {
            diag_obs(ell);
			P = matpow(D * transition.transpose(), p);
        }
		ahat = P * ahat;
        c(ell) = ahat.sum();
        if (isnan(toDouble(c(ell))) || c(ell) <= 0.0)
            throw std::domain_error("nan encountered in hmm");
        ahat /= c(ell);
    }
}

template <typename T>
void HMM<T>::forward(void)
{
    D.setZero();
    diag_obs(0);
	c = Vector<T>::Zero(Ltot);
    alpha_hat.col(0) = D * pi;
	c(0) = 1.0;
	Matrix<T> P;
	int lt = 1;
    for (int ell = 0; ell < L; ++ell)
    {
        int R = obs(ell, 0);
		if (ell == 0)
			R--;
		diag_obs(ell);
		P = D * transition.transpose();
		for (int r = 0; r < R; ++r)
		{
			alpha_hat.col(lt) = P * alpha_hat.col(lt - 1);
			c(lt) = alpha_hat.col(lt).sum();
			if (isnan(toDouble(c(lt - 1))) || c(lt - 1) <= 0.0)
				throw std::domain_error("nan encountered in hmm");
			alpha_hat.col(lt) /= c(lt);
			++lt;
		}
    }
	if (Ltot != lt)
		throw std::domain_error("something went wrong");
}

template <typename T>
void HMM<T>::backward(void)
{
    beta_hat.col(Ltot - 1) = Vector<T>::Ones(M);
	Matrix<T> P;
	int R;
	int lt = Ltot - 2;
    for (int ell = L - 1; ell >= 0; --ell)
	{
		R = obs(ell, 0);
		diag_obs(ell);
		P = transition * D;
		for (int r = 0; r < R; ++r)
		{
			beta_hat.col(lt) = P * beta_hat.col(lt + 1) / c(lt + 1);
			--lt;
		}
	}
	if (lt != -1)
		throw std::domain_error("something went wrong");
}

template <typename T>
Matrix<T> HMM<T>::xi(int ell)
{
	diag_obs(ell);
	Matrix<T> M = c(ell) * alpha_hat.col(ell - 1) * beta_hat.col(ell).transpose() * D;
	return M.cwiseProduct(transition);
}

template <typename T>
T HMM<T>::Q(void)
{
	forward();
	backward();
	Matrix<T> log_transition = transition.array().log();
	Matrix<T> gamma = alpha_hat.cwiseProduct(beta_hat);
	T ret = gamma.col(0).transpose() * pi.array().log().matrix();
	for (int ell = 1; ell < Ltot; ++ell)
		ret += xi(ell).cwiseProduct(log_transition).sum();
	for (int ell = 0; ell < Ltot; ++ell)
	{
		diag_obs(ell);
		ret += (D * gamma.col(ell)).sum();
	}
	return ret;
}


template <typename T>
T compute_hmm_likelihood(
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
        results.emplace_back(tp.enqueue([&] { return hmm.Q(); }));
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
        const std::vector<Matrix<double>>& emission, 
        const int L, const std::vector<int*> obs,
        int numthreads, 
        bool viterbi, std::vector<std::vector<int>> &viterbi_paths);

template adouble compute_hmm_likelihood(
        const Vector<adouble> &pi, const Matrix<adouble> &transition,
        const std::vector<Matrix<adouble>>& emission, 
        const int L, const std::vector<int*> obs,
        int numthreads, 
        bool viterbi, std::vector<std::vector<int>> &viterbi_paths);

template class HMM<double>;
template class HMM<adouble>;

void print_matrix(Matrix<adouble> &M) { std::cout << M.cast<double>() << std::endl << std::endl; }

