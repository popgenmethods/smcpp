#include "transition_bundle.h"

void TransitionBundle::update(const Matrix<adouble> &new_T, const bool recompute_eigs)
{
    T = new_T;
    Td = T.template cast<double>();
    if (! recompute_eigs) return;
    eigensystems.clear();
    const int M = T.rows();
    span_Qs.clear();
    Matrix<double> tmp;

    eigensystems.clear();
    for (auto it = targets.begin(); it != targets.end(); ++it)
    {
        block_key key = it->second;
        if (this->eigensystems.count(key) == 0)
        {
            Vector<double> ep = this->emission_probs->at(key).template cast<double>();
            tmp = ep.template cast<double>().asDiagonal() * this->Td.transpose();
            DEBUG1 << key << "\n" << ep.transpose();
            Eigen::EigenSolver<Matrix<double> > es(tmp);
            this->eigensystems.emplace(key, es);
        }
    }

//#pragma omp parallel
//#pragma omp single
    for (auto it = targets.begin(); it != targets.end(); ++it)
    {
// #pragma omp task firstprivate(it)
        {
            const int span = it->first;
            const block_key key = it->second;
            const eigensystem eig = this->eigensystems.at(key);
            Matrix<double> Q(M, M);
            for (int a = 0; a < M; ++a)
            {
                double d1 = eig.d_r_scaled(a);
                Q(a, a) = std::pow(d1, span - 1) * (double)span;
                for (int b = a + 1; b < M; ++b)
                {
                    d1 = eig.d_r_scaled(a);
                    double d2 = eig.d_r_scaled(b);
                    if (std::abs(d1) < std::abs(d2))
                        std::swap(d1, d2);
                    Q(a, b) = std::exp(
                            (double)span * std::log(d1) + std::log1p(-std::pow(d2 / d1, span))
                            );
                    Q(a, b) /= d1 - d2;
                    Q(b, a) = Q(a, b);
                }
            }
//#pragma omp critical(emplace_Q)
            {
                this->span_Qs.emplace(*it, Q);
            }
        }
    }
//#pragma omp taskwait
}
