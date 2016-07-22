#include "conditioned_sfs.h"

struct jcsfs_eigensystem
{
    eigensystem(const Matrix<double> &M) : es(M, true), 
        U(es.eigenvectors().real()), D(es.eigenvalues().real()),
        Uinv(U.inv()) {}
    const Eigen::Eigensolver<Matrix<double> >;
    const Matrix<double> U, Uinv;
    const Vector<double> eig;

    template <typename T>
    const expM(T t)
    {
        return U.template cast<T>() * (t * D.asDiagonal().template cast<T>()) * Uinv.template cast<T>();
    }
};

template <typename T>
class JointCSFS
{
    public:
    JointCSFS(int n1, int n2, int a1, int a2, std::vector<double> hidden_states) : 
        n1(n1), n2(n2), a1(a1), a2(a2),
        csfs_n1(n1),
        csfs_n1n2(n1 + n2 - 2),
        Mn1(moran_rate_matrix(n1 + 1)),
        Mn2(moran_rate_matrix(n2)),
        Mn10(modified_moran_rate_matrix(n1, 0)),
        Mn11(modified_moran_rate_matrix(n1, 1))
        J(hidden_states.size() - 1, a1 + 1, n1 + 1, a2 + 1, n2 + 1)
        {}

    void compute(const ParameterVector&, const ParameterVector&, const double);
     
    private:
    Vector<T> undistinguishedSFS(const ParameterVector &params, const double t1, const double t2) const;
    // Member variables
    const OnePopConditionedSFS<T> csfs_n1, csfs_n1n2;
    const jcsfs_eigensystem Mn1, Mn2, Mn10, Mn11;
    Eigen::Tensor<T, 5, Eigen::RowMajor> > J;
}

template <typename T>
Vector<T> jcsfs_helper_tau_above_split(const int m, const double split, 
        const ParameterVector &model1, const ParameterVector& model,
        T Rts1, T Rts2, int n1, int n2, int a1, int a2)
{
    // Returns a flattened version of a 3 * (n1 + 1) * (n2 + 1) tensor
    double t1 = hidden_states[m], t2 = hidden_states[m + 1];
    assert(split <= t1 < t2);
    assert(a1 == 2);

    // Shift eta1 back by split units in time 
    PiecewiseConstantRateFunction<T> shiftedEta1 = shiftRateFunction(model1, split);
    Vector<T> rsfs = undistinguishedSFS(shiftedEta1, n1 + n2, t1 - split, t2 - split);
    Matrix<T> eMn1[3];
    eMn1[0] = Mn10.expM(Rts1);
    eMn1[1] = Mn11.expM(Rts1);
    eMn1[2] = eMn1[0].reverse();
    Matrix<T> eMn2 = Mn2.expM(Rts2)}

    for (int b1 = 0; b1 < n1 + 1; ++b1)
        for (int b2 = 0; b2 < n2 + 1; ++b2)
            for (int nseg = 0; nseg < n1 + n2 + 1; ++nseg)
                for (int np1 = std::max(nseg - n2, 0), nseg < tsd::min(nseg, n1) + 1; ++nseg)
                {
                    int np2 = nseg - np1;
                    // scipy.stats.hypergeom.pmf(np1, n1 + n2, nseg, n1)  = choose(nseg, np1) * choose(n1 + n2 - nseg, n1 - np1)
                    // gsl_ran_hypergeometric_pdf(unsigned int k, unsigned int n1, unsigned int n2, unsigned int t) 
                    double h = gsl_ran_hypergeometric_pdf(np1, nseg, n1 + n2 - nseg, n1);
                    for (int i = 0; i < 3; ++i)
                        J(m, i, b1, 0, b2) += h * rsfs(i, nseg) * eMn1[i](np1, b1) * eMn2(np2, b2);
                }
     
    PiecewiseConstantRateFunction<T> eta1(model1, {split - 1e-6, split + 1e-6});
    Matrix<T> sfs_below = csfs_n1.compute(eta)[0];
    for (int i = 0; i < a1 + 1; ++i)
        for (int j = 0; j < n1 + 1; ++j)
            J(m, i, j, 0, 0) += sfs_below(i, j);
}

template <typename T>
Vector<T> JointCSFS<T>::undistinguishedSFS(const ParameterVector &params, const int t1, const int t2) const
{
    PiecewiseConstantRateFunction<T> eta(params, {t1, t2});
    Matrix<T> csfs = csfs_n1n2.compute(eta)[0];
    Vector<T> unsfs(n1 + n2 - 1);
    unsfs.setZero();
    for (int a = 0; a < 2; ++a)
        for (int b = 0; b < n1 + n2 + 1; ++b)
            if (1 <= a + b <= n1 + n2 + 1)
                ret(a + b - 1) += csfs(a, b);
    return ret;
}

template <typename T>
PiecewiseConstantRateFunction<T> shiftRateFunction(const ParameterVector &model1, double shift)
{
    std::vector<T> s = model1[1];
    T zero = 0 * model1[0][0];
    std::vector<T> cs(s.size() + 1);
    cs[0] = zero;
    std::partial_sum(s.begin(), s.end(), cs.begin() + 1);
    T tshift = zero + shift;
    int ip = insertion_point(tshift, cs, 0, cs.size());
    cs[ip] = tshift;
    std::vector<T> sp;
    int i = ip;
    while (i + 1 < cs.size())
        sp.push_back(zero + (cs[ip + 1] - cs[ip]));
    std::vector<T> ap(a.start() + ip, a.end());
    return PiecewiseConstantRateFunction<T>({ap, sp}, eta.hidden_states);
}
