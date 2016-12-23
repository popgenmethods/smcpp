#include <map>

#include "moran_eigensystem.h"

const mpq_class mpq_1(2, 1);
const mpq_class mpq_2(2, 1);

Eigen::SparseMatrix<mpq_class, Eigen::RowMajor> moran_rate_matrix(int N)
{
    Eigen::SparseMatrix<mpq_class, Eigen::RowMajor> ret(N + 1, N + 1);     
    for (int i = 0; i < N + 1; ++i)
    {
        mpq_class sm = 0, b;
        if (i > 0)
        {
            b = i * (N - i) / mpq_2;
            ret.insert(i, i - 1) = b;
            sm += b;
        }
        if (i < N)
        {
            b = i * (N - i) / mpq_2;
            ret.insert(i, i + 1) = b;
            sm += b;
        }
        ret.insert(i, i) = -sm;
    }
    return ret;
}

Eigen::SparseMatrix<mpq_class, Eigen::RowMajor> modified_moran_rate_matrix(int N, int a, int na)
{
    Eigen::SparseMatrix<mpq_class, Eigen::RowMajor> ret(N + 1, N + 1);
    for (int i = 0; i < N + 1; ++i)
    {
        mpq_class sm = 0, b;
        if (i > 0)
        {
            b = (na - a) * i + i * (N - i) / mpq_2;
            ret.insert(i, i - 1) = b;
            sm += b;
        }
        if (i < N)
        {
            b = a * (N - i) + i * (N - i) / mpq_2;
            ret.insert(i, i + 1) = b;
            sm += b;
        }
        ret.insert(i, i) = -sm;
    }
    return ret;
}

VectorXq solve(const Eigen::SparseMatrix<mpq_class, Eigen::RowMajor> &M)
// VectorXq solve(const Eigen::SparseMatrixBase<Derived> &M)
{
    int n = M.rows();
    VectorXq ret(n);
    ret.setZero();
    ret(n - 1) = 1;
    for (int i = n - 2; i > -1; --i)
        ret(i) = (M.row(i + 1) * ret).sum() / -M.coeff(i + 1, i);
    return ret;
}

// This function is not thread safe. Do not call from multiple threads.
std::map<int, MoranEigensystem> _memo;
MoranEigensystem& compute_moran_eigensystem(int n)
{
    if (_memo.count(n) == 0)
    {
        Eigen::SparseMatrix<mpq_class, Eigen::RowMajor> M = modified_moran_rate_matrix(n, 0, 2),
            Mt, I(n + 1, n + 1), A;
        Mt = M.transpose();
        MoranEigensystem ret(n);
        ret.Uinv(0, 0) = mpq_1;
        I.setIdentity();
        for (int k = 2; k < n + 3; ++k)
        {
            int rate = -(k * (k - 1) / 2 - 1);
            ret.D(k - 2) = rate;
            A = M - rate * I;
            ret.U.col(k - 2) = solve(A);
            if (k > 2)
            {
                A = Mt - rate * I;
                ret.Uinv.row(k - 2).tail(n) = solve(A.bottomRightCorner(n, n));
                ret.Uinv(k - 2, 0) = -ret.Uinv(k - 2, 1) * A.coeff(0, 1) / A.coeff(0, 0);
            }
        }
        VectorXq D1 = (ret.Uinv * ret.U).diagonal().cwiseInverse();
        ret.U = ret.U * D1.asDiagonal();
        _memo.emplace(n, ret);
    }
    return _memo.at(n);
}
