#include <fstream>
#include <cereal/cereal.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/map.hpp>

#include "matrix_cache.h"
#include "moran_eigensystem.h"
#include "mpq_support.h"

namespace Eigen
{
    template <class Archive, class _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols> inline
    void
    save(Archive & ar, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> const & m)
    {
        int32_t rows = m.rows();
        int32_t cols = m.cols();
        ar(rows);
        ar(cols);
        ar(cereal::binary_data(m.data(), rows * cols * sizeof(_Scalar)));
    }

    template <class Archive, class _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols> inline
        void
    load(Archive & ar, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> & m)
    {
        int32_t rows;
        int32_t cols;
        ar(rows);
        ar(cols);

        m.resize(rows, cols);

        ar(cereal::binary_data(m.data(), static_cast<std::size_t>(rows * cols * sizeof(_Scalar))));
    }
}

static std::map<int, MatrixCache> cache;
static std::string store_location;

void init_cache(const std::string loc)
{
    store_location = loc;
    std::ifstream in(store_location);
    if (in)
    {
        cereal::BinaryInputArchive iarchive(in);
        iarchive(cache);
    }
}

void store_cache()
{
    DEBUG << "storing cache: " << store_location;
    std::ofstream out(store_location);
    if (out)
    {
        cereal::BinaryOutputArchive oarchive(out);
        oarchive(cache);
    }
    else
        ERROR << "could not open cache file for storage";
}

typedef struct { MatrixXq coeffs; } below_coeff;

std::map<int, below_coeff> below_coeffs_memo;
below_coeff compute_below_coeffs(int n)
{
    if (below_coeffs_memo.count(n) == 0)
    {
        DEBUG << "Computing below_coeffs";
        below_coeff ret;
        MatrixXq mlast;
        for (int nn = 2; nn < n + 3; ++nn)
        {
            MatrixXq mnew(n + 1, nn - 1);
            mnew.col(nn - 2).setZero();
            mnew(nn - 2, nn - 2) = 1;
#pragma omp parallel for
            for (int k = nn - 1; k > 1; --k)
            {
                long denom = (nn + 1) * (nn - 2) - (k + 1) * (k - 2);
                mpq_class c1((nn + 1) * (nn - 2), denom);
                mnew.col(k - 2) = mlast.col(k - 2) * c1;
            }
            for (int k = nn - 1; k > 1; --k)
            {
                long denom = (nn + 1) * (nn - 2) - (k + 1) * (k - 2);
                mpq_class c2((k + 2) * (k - 1), denom);
                mnew.col(k - 2) -= mnew.col(k - 1) * c2;
            }
            mlast = mnew;
        }
        ret.coeffs = mlast;
        below_coeffs_memo.emplace(n, ret); 
    }
    return below_coeffs_memo.at(n);
}

std::map<std::array<int, 3>, mpq_class> _Wnbj_memo;
mpq_class calculate_Wnbj(int n, int b, int j)
{
    switch (j)
    {
        case 2:
            return mpq_class(6, n + 1);
        case 3:
            if (n == 2 * b) return 0;
            return mpq_class(30 * (n - 2 * b), (n + 1) * (n + 2));
        default:
            std::array<int, 3> key = {n, b, j};
            if (_Wnbj_memo.count(key) == 0)
            {
                int jj = j - 2;
                mpq_class c1(-(1 + jj) * (3 + 2 * jj) * (n - jj), jj * (2 * jj - 1) * (n + jj + 1));
                mpq_class c2((3 + 2 * jj) * (n - 2 * b), jj * (n + jj + 1));
                mpq_class ret = calculate_Wnbj(n, b, jj) * c1;
                ret += calculate_Wnbj(n, b, jj + 1) * c2;
                _Wnbj_memo[key] = ret;
            }
            return _Wnbj_memo[key];
    }
}

std::map<std::array<int, 3>, mpq_class> pnkb_dist_memo;
mpq_class pnkb_dist(int n, int m, int l1)
{
    // Probability that lineage 1 has size |L_1|=l1 below tau,
    // the time at which 1 and 2 coalesce, when there are k 
    // undistinguished lineages remaining, in a sample of n
    // undistinguished (+2 distinguished) lineages overall.
    std::array<int, 3> key{n, m, l1};
    if (pnkb_dist_memo.count(key) == 0)
    {
        mpz_class binom1, binom2;
        mpz_bin_uiui(binom1.get_mpz_t(), n + 2 - l1, m + 1);
        mpz_bin_uiui(binom2.get_mpz_t(), n + 3, m + 3);
        mpq_class ret(binom1, binom2);
        ret *= l1;
        pnkb_dist_memo[key] = ret;
    }
    return pnkb_dist_memo[key];
}

std::map<std::array<int, 3>, mpq_class> pnkb_undist_memo;
mpq_class pnkb_undist(int n, int m, int l3)
{
    // Probability that undistinguished lineage has size |L_1|=l1 below tau,
    // the time at which 1 and 2 coalesce, when there are k 
    // undistinguished lineages remaining, in a sample of n
    // undistinguished (+2 distinguished) lineages overall.
    std::array<int, 3> key{n, m, l3};
    if (pnkb_undist_memo.count(key) == 0)
    {
        mpz_class binom1, binom2;
        mpz_bin_uiui(binom1.get_mpz_t(), n + 3 - l3, m + 2);
        mpz_bin_uiui(binom2.get_mpz_t(), n + 3, m + 3);
        mpq_class ret(binom1, binom2);
        pnkb_undist_memo.emplace(key, ret);
    }
    return pnkb_undist_memo[key];
}

template <typename Derived1, typename Derived2, typename Derived3>
void parallel_matmul(const Eigen::DenseBase<Derived1>& A, 
        const Eigen::DenseBase<Derived2>& B, 
        const Eigen::DenseBase<Derived3>& C, 
        Matrix<double> &dst)
{
    // Compute A * diag(B) * C and store in dst. 
    // Eigen won't parallelize MatrixXq multiplication. 
    int m = A.rows();
    int n = C.cols();
    DEBUG << m << " " << n;
    dst.resize(m, n);
    MatrixXq tmp(m, n);
    tmp.setZero();
#pragma omp parallel for collapse(2)
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < B.size(); ++k)
                tmp(i, j) += A(i, k) * B(k) * C(k, j);
#pragma omp parallel for collapse(2)
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            dst(i, j) = tmp(i, j).get_d();
}

MatrixCache& cached_matrices(const int n)
{
    if (cache.count(n) == 0)
    {
        DEBUG << "moran eigensystem";
        const MoranEigensystem mei = compute_moran_eigensystem(n);
        DEBUG << "moran eigensystem done";
        VectorXq D_subtend_above = VectorXq::LinSpaced(n, 1, n);
        D_subtend_above /= n + 1;

        VectorXq D_subtend_below = Eigen::Array<mpq_class, Eigen::Dynamic, 1>::Ones(n + 1) / 
            Eigen::Array<mpq_class, Eigen::Dynamic, 1>::LinSpaced(n + 1, 2, n + 2);
        D_subtend_below *= 2;

        MatrixXq Wnbj(n, n), P_dist(n + 1, n + 1), P_undist(n + 1, n);
        Wnbj.setZero();
        for (int b = 1; b < n + 1; ++b)
            for (int j = 2; j < n + 2; ++j)
                Wnbj(b - 1, j - 2) = calculate_Wnbj(n + 1, b, j);

        // P_dist(k, b) = probability of state (1, b) when there are k undistinguished lineages remaining
        P_dist.setZero();
        for (int k = 0; k < n + 1; ++k)
            for (int b = 1; b < n - k + 2; ++b)
                P_dist(k, b - 1) = pnkb_dist(n, k, b);

        // P_undist(k, b) = probability of state (0, b + 1) when there are k undistinguished lineages remaining
        P_undist.setZero();
        for (int k = 1; k < n + 1; ++k)
            for (int b = 1; b < n - k + 2; ++b)
                P_undist(k, b - 1) = pnkb_undist(n, k, b);

        VectorXq lsp = VectorXq::LinSpaced(n + 1, 2, n + 2);

        DEBUG << "Eigen uses " << Eigen::nbThreads() << " for matmul";

        below_coeff bc = compute_below_coeffs(n);

        // This is too slow. 
        DEBUG << "X0";
        parallel_matmul(Wnbj.transpose(), 
                VectorXq::Ones(n) - D_subtend_above, 
                mei.U.bottomRows(n), 
                cache[n].X0);
        // ret.X0 = (Wnbj.transpose() * (VectorXq::Ones(n) - D_subtend_above).asDiagonal() * 
        //        mei.U.bottomRows(n)).template cast<double>();
        DEBUG << "X2";
        parallel_matmul(Wnbj.transpose(), 
                D_subtend_above, 
                mei.U.reverse().topRows(n),
                cache[n].X2);
        // ret.X2 = (Wnbj.transpose() * D_subtend_above.asDiagonal() * 
        //         mei.U.reverse().topRows(n)).template cast<double>();
        DEBUG << "M0";
        parallel_matmul(bc.coeffs,
                lsp.cwiseProduct((VectorXq::Ones(n + 1) - D_subtend_below)),
                P_undist,
                cache[n].M0);
        // ret.M0 = (bc.coeffs * lsp.asDiagonal() * (VectorXq::Ones(n + 1) - 
        //             D_subtend_below).asDiagonal() * P_undist).template cast<double>();
        DEBUG << "M1";
        parallel_matmul(bc.coeffs,
                lsp.cwiseProduct(D_subtend_below),
                P_dist,
                cache[n].M1);
        store_cache();
    }
    return cache.at(n);
}
