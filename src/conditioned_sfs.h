#ifndef CONDITIONED_SFS_H
#define CONDITIONED_SFS_H

#include <cassert>
#include <cfenv>
#include <map>
#include <thread>

#include "common.h"
#include "piecewise_exponential.h"
#include "prettyprint.hpp"

class ConditionedSFS
{
    public:
    ConditionedSFS(PiecewiseExponential, int);
    void compute(int, int, const std::vector<double>&, 
            const std::vector<Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>> &,
            double, double);
    std::thread compute_threaded(int, int, const std::vector<double>&, 
            const std::vector<Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>> &,
            double, double);
    AdMatrix matrix() const;

    private:
    // Methods
    void fill_matrices();
    void construct_ad_vars();
    void compute_ETnk_below(const AdVector&);
    double exp1();
    double exp1_conditional(double, double);
    double unif();

    // Variables
    std::mt19937 gen;
    PiecewiseExponential eta;
    const int n;
    Eigen::MatrixXd D_subtend_above, D_not_subtend_above, D_subtend_below, D_not_subtend_below, Wnbj, P_dist, P_undist, tK;
    AdMatrix csfs, csfs_above, csfs_below, ETnk_below;
};

void store_sfs_results(const AdMatrix&, double*, double*);
AdMatrix calculate_sfs(PiecewiseExponential eta, int n, int S, int M, const std::vector<double> &ts, 
        const std::vector<double*> &expM, double t1, double t2, int numthreads, double theta);
void set_seed(long long);

#endif
