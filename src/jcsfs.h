#ifndef JCSFS_H
#define JCSFS_H

#include <Eigen/Eigenvalues>

#include "common.h"
#include "conditioned_sfs.h"
#include "moran_eigensystem.h"
#include "mpq_support.h"

// This is a line-by line translation of the smcpp/jcsfs.py. See that
// file for motivating comments.
template <typename T>
class JointCSFS : public ConditionedSFS<T>
{
    public:
    JointCSFS(int n1, int n2, int a1, int a2, const std::vector<double> hidden_states) : 
        hidden_states(hidden_states),
        M(hidden_states.size() - 1),
        K(10), // number of random trials used to compute transition matrices.
        n1(n1), n2(n2), a1(a1), a2(a2), csfs(make_csfs()),
        Mn1(moran_rate_matrix(n1 + 1)),
        Mn2(moran_rate_matrix(n2)),
        Mn10(modified_moran_rate_matrix(n1, 0)),
        Mn11(modified_moran_rate_matrix(n1, 1)),
        Mn12(modified_moran_rate_matrix(n1, 2)),
        S2(make_S2()),
        S0(Vector<double>::Ones(n1 + 2) - S2),
        J(M, Matrix<T>::Zero(a1 + 1, (n1 + 1) * (n2 + 1)))
        {}

    // This method exists for compatibility with the parent class interface. 
    std::vector<Matrix<T> > compute(const PiecewiseConstantRateFunction<T>&) const;
    // This method actually does the work and must be called before compute().
    void pre_compute(const ParameterVector&, const ParameterVector&, const double);
     
    private:
    struct jcsfs_eigensystem
    {
        jcsfs_eigensystem(const Matrix<mpq_class> &M) : 
            Md(M.template cast<double>()),
            es(Md, true), 
            U(es.eigenvectors().real()), Uinv(U.inverse()),
            D(es.eigenvalues().real()) {}
        const Matrix<double> Md;
        const Eigen::EigenSolver<Matrix<double> > es;
        const Matrix<double> U, Uinv;
        const Vector<double> D;

        const Matrix<T> expM(T t) const
        {
            Vector<T> eD = (t * D.template cast<T>()).array().exp().matrix();
            Matrix<T> ret = U.template cast<T>() * eD.asDiagonal() * Uinv.template cast<T>();
            return ret;
        }
    };
 
    // Private functions
    inline T& tensorRef(const int m, const int i, const int j, const int k) 
    { 
        int ind = j * (n2 + 1) + k;
        return J[m].coeffRef(i, ind);
    }

    std::map<int, OnePopConditionedSFS<T> > make_csfs();
    Vector<double> make_S2();
    void jcsfs_helper_tau_above_split(const int, const double, const double, const T);
    void jcsfs_helper_tau_below_split(const int, const double, const double, const T);

    // Member variables
    const std::vector<double> hidden_states;
    const int M, K;
    const int n1, n2, a1, a2;
    const std::map<int, OnePopConditionedSFS<T> > csfs;
    const jcsfs_eigensystem Mn1, Mn2, Mn10, Mn11, Mn12;
    const Vector<double> S2, S0;

    // These change at each call of compute
    std::vector<Matrix<T> > J;
    double split;
    T Rts1, Rts2;
    ParameterVector params1, params2;
    std::unique_ptr<PiecewiseConstantRateFunction<T> > eta1, eta2;
    std::array<Matrix<T>, 3> eMn1;
    Matrix<T> eMn2;
};

#endif
