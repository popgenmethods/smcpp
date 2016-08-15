#ifndef MATRIX_CACHE_H
#define MATRIX_CACHE_H

#include "common.h"

struct MatrixCache
{ 
    template <class Archive>
    void serialize(Archive & ar)
    {
        ar(X0, X2, M0, M1);
    }
    Matrix<double> X0, X2, M0, M1; 
};

MatrixCache& cached_matrices(int);
void init_cache(const std::string);

#endif
