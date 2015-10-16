#ifndef EXPONENTIAL_INTEGRALS_H
#define EXPONENTIAL_INTEGRALS_H

#include "gsl/gsl_sf_expint.h"
#include "common.h"

template <typename T>
T eintdiff(const T&, const T&, const T&);

#endif
