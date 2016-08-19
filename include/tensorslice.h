#ifndef TENSORSLICE_H
#define TENSORSLICE_H

#include <Eigen/Dense>

template <size_t D>
struct tensorSlice
{
    template <typename Derived1, typename Derived2, typename Derived3>
    static Eigen::Matrix<typename Derived1::Scalar, Eigen::Dynamic, 1> run(
            const Eigen::DenseBase<Derived1> &tensor,
            const Eigen::DenseBase<Derived2> &inds,
            const Eigen::DenseBase<Derived3> &dims);
};

template <>
template <typename Derived1, typename Derived2, typename Derived3>
Eigen::Matrix<typename Derived1::Scalar, Eigen::Dynamic, 1> tensorSlice<1>::run(
        const Eigen::DenseBase<Derived1> &tensor,
        const Eigen::DenseBase<Derived2> &inds,
        const Eigen::DenseBase<Derived3> &dims)
{
    assert(dims.size() == 1);
    assert(inds.size() == 1);
    assert(tensor.cols() == dims(0));
    return tensor.col(inds(0));
}

template <size_t D>
template <typename Derived1, typename Derived2, typename Derived3>
Eigen::Matrix<typename Derived1::Scalar, Eigen::Dynamic, 1> tensorSlice<D>::run(
        const Eigen::DenseBase<Derived1> &tensor,
        const Eigen::DenseBase<Derived2> &inds,
        const Eigen::DenseBase<Derived3> &dims)
{
    const int slice = dims.template tail<D - 1>().prod();
    return tensorSlice<D - 1>::run(
        tensor.middleCols(inds(0) * slice, slice),
        inds.template tail<D - 1>(),
        dims.template tail<D - 1>()
    );
}

#endif
