#ifndef CONDITIONED_SFS_H
#define CONDITIONED_SFS_H

#include <cassert>
#include <cfenv>
#include <map>

#include "common.h"
#include "piecewise_exponential.h"
#include "prettyprint.hpp"


class ConditionedSFS
{
    public:
    ConditionedSFS(PiecewiseExponential*, int);
    void compute(int, int, double*, std::vector<double*>, int*, double*);
    void store_results(double*, double*);
    AdMatrix& matrix();

    private:
    // Methods
    void fill_matrices();
    void construct_ad_vars();
    void compute_ETnk_below(const AdVector&);

    // Variables
    PiecewiseExponential *eta;
    const int n;
    Eigen::MatrixXd D_subtend_above, D_not_subtend_above, D_subtend_below, D_not_subtend_below, Wnbj, P_dist, P_undist, tK;
    AdMatrix csfs, csfs_above, csfs_below, ETnk_below;
};

#endif
