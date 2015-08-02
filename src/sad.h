#include <Eigen/Dense>

namespace sad
{
    template <typename T>
    struct simple_autodiff
    {
        typedef Eigen::Matrix<T, 1, 1> Scalar;
        typedef Eigen::Matrix<T, Eigen::Dynamic, 1> Vec;
        Eigen::Ref<Scalar> x;
        Eigen::Ref<Vec> d;
    };

    template <typename T>
    simple_autodiff<T> operator+(const simple_autodiff<T> &s1, const simple_autodiff<T> &s2)
    {
        simple_autodiff<T> ret;
        ret.x = s1.x + s2.x;
        ret.d = s1.d + s2.d;
        return ret;
    }
}
