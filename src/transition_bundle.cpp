#include "transition_bundle.h"

std::mutex checkEigensytem_mtx;
std::mutex spanQinsert_mtx;
void TransitionBundle::update(const Matrix<adouble> &new_T)
{
    T = new_T;
    Td = T.template cast<double>();
    const int M = T.rows();
    eigensystems.clear();
    span_Qs.clear();
    std::vector<std::future<void> > results;
    for (auto it = targets.begin(); it < targets.end(); ++it)
        results.emplace_back(tp.enqueue([it, this, M]
        {
            Matrix<double> tmp;
            int span = it->first;
            block_key key = it->second;
            if (this->eigensystems.count(key) == 0)
            {
                std::lock_guard<std::mutex> lock(checkEigensytem_mtx);
                tmp = this->emission_probs->at(key).template cast<double>().asDiagonal() * this->Td.transpose();
                Eigen::EigenSolver<Matrix<double> > es(tmp);
                this->eigensystems.emplace(key, es);
            }
            eigensystem eig = this->eigensystems.at(key);
            Matrix<std::complex<double> > Q(M, M);
            for (int a = 0; a < M; ++a)
                for (int b = 0; b < M; ++b)
                    if (a == b)
                        Q(a, b) = std::pow(eig.d_scaled(a), span - 1) * (double)span;
                    else
                        Q(a, b) = (std::pow(eig.d_scaled(a), span) - std::pow(eig.d_scaled(b), span)) / 
                            (eig.d_scaled(a) - eig.d_scaled(b));
            {
                std::lock_guard<std::mutex> lock(spanQinsert_mtx);
                this->span_Qs.emplace(*it, Q);
            }
        }));
    for (auto &&result : results)
        result.wait();
}
