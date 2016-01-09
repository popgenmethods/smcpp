#ifndef TRANSITION_BUNDLE_H
#define TRANSITION_BUNDLE_H

struct eigensystem
{
    Eigen::MatrixXcd P, Pinv;
    Eigen::VectorXcd d;
    eigensystem(const Eigen::EigenSolver<Matrix<double> > &es) :
        P(es.eigenvectors()), Pinv(P.inverse()), d(es.eigenvalues()) {}
};

class TransitionBundle
{
    public:
    TransitionBundle(const std::set<std::pair<int, block_key> > &targets,
            const std::map<block_key, Vector<adouble> >* emission_probs) : 
        targets(targets), emission_probs(emission_probs) {}

    void update(const Matrix<adouble> &new_T)
    {
        T = new_T;
        Td = T.template cast<double>();
        Matrix<double> tmp;
        int M = T.rows();
        Eigen::MatrixXcd A(M, M);
        eigensystems.clear();
        span_Qs.clear();
        for (auto &p : targets)
        {
            int span = p.first;
            block_key key = p.second;
            if (eigensystems.count(key) == 0)
            {
                tmp = emission_probs->at(key).template cast<double>().asDiagonal() * Td.transpose();
                es = Eigen::EigenSolver<Matrix<double> >(tmp);
                eigensystems.emplace(key, es);
            }
            eigensystem eig = eigensystems.at(key);
            for (int a = 0; a < M; ++a)
                for (int b = 0; b < M; ++b)
                    A(a, b) = (a == b) ? (double)span * std::pow(eig.d(a), span - 1) : 
                        (std::pow(eig.d(a), span) - std::pow(eig.d(b), span)) / (eig.d(a) - eig.d(b));
            span_Qs[{span, key}] = A;
        }
    }
    Matrix<adouble> T;
    Matrix<double> Td;
    Eigen::EigenSolver<Matrix<double> > es;
    Eigen::VectorXcd d;
    Eigen::MatrixXcd P, Pinv;
    std::map<std::pair<int, block_key>, Eigen::MatrixXcd> span_Qs;
    std::map<block_key, eigensystem> eigensystems;

    private:
    const std::set<std::pair<int, block_key> > targets;
    const std::map<block_key, Vector<adouble> >* emission_probs;
};

#endif

