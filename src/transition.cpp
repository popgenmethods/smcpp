#include "transition.h"

template <typename T>
struct integrand_helper
{
    const int i, j;
    const PiecewiseExponentialRateFunction<T> *eta;
};

template <typename T>
T integrand(T x, T y, void* ih)
{
    // h=t[i - 1]..t[i]
    // u=0..h
    integrand_helper<T> *_ih = (integrand_helper<T>*)ih;
    int i = _ih->i, j = _ih->j; 
    const PiecewiseExponentialRateFunction<T> *eta = _ih->eta;
    const std::vector<double> &t = eta->hidden_states;
    double h, u, jac, dx = toDouble(x), dy = toDouble(y);
    if (t[i] < INFINITY)
    {
        h = t[i] * dx + t[i - 1] * (1. - dx);
        u = dy * h;
        jac = (t[i] - t[i - 1]) * h;
    }
    else
    {
        dx = std::min(dx, .9999999);
        h = t[i - 1] + dx / (1. - dx);
        u = dy * h;
        jac = h / (1. - u) / (1. - u);
    }
    T ret = 0.0, a, b;
    double left = t[j - 1];
    double right = t[j];
    // i1
    a = -2 * (eta->R(h) - eta->R(u)) - eta->R(std::max(h, left));
    if (right == INFINITY)
        ret += exp(a);
    else
    {
        b = -2 * (eta->R(h) - eta->R(u)) - eta->R(std::max(h, right));
        if (a != b)
            ret += exp(a) - exp(b);
    }
    // i2
    a = 2 * eta->R(u) - eta->R(h) - 2 * eta->R(std::max(u, std::min(left, h)));
    b = 2 * eta->R(u) - eta->R(h) - 2 * eta->R(std::max(u, std::min(right, h)));
    if (a != b)
        ret += 0.5 * (exp(a) - exp(b));
    // i3
    if (i == j)
        ret += 0.5 * exp(-eta->R(h)) * -expm1(-2 * (eta->R(h) - eta->R(u)));
    ret *= eta->eta(h) / h;
    ret *= jac;
    return ret;
}

template <typename T>
T Transition<T>::trans(int i, int j)
{
    const std::vector<double> t = eta->hidden_states;
    integrand_helper<T> ih = {i, j, eta};
    T num = gauss_legendre_2D_cube<T>(128, integrand<T>, (void*)&ih, (T)0., (T)1., (T)0., (T)1.);
    T denom = exp(-eta->R(t[i - 1]));
    if (t[i] != INFINITY)
        denom -= exp(-eta->R(t[i]));
    return num / denom;
}

template <typename T>
struct p_intg_helper
{
    const PiecewiseExponentialRateFunction<T>* eta;
    const double rho;
};

template <typename T>
T p_integrand(T t, void *h)
{
    p_intg_helper<T> *hh = (p_intg_helper<T>*)h;
    T e1 = hh->eta->R(t);
    T e2 = t * hh->rho;
    T e3 = e1 + e2;
    return hh->eta->eta(t) * exp(-e3);
}

template <typename T>
T Transition<T>::P_no_recomb(const int i, const double rho)
{
    p_intg_helper<T> h = {eta, rho};
    std::vector<double> t = eta->hidden_states;
    T left = t[i - 1];
    T right = std::min(t[i], 100.);
    T ret = gauss_legendre<T>(128, &p_integrand<T>, (void*)&h, left, right);
    T denom = exp(-eta->R(t[i - 1]));
    if (t[i] != INFINITY)
        denom -= exp(-eta->R(t[i]));
    ret /= denom;
    return ret;
}

/*
template <typename T>
QuadPoints Transition<T>::make_quad_points()
{
    const int rule = 20;
    int order_num = dunavant_order_num(rule);
    double* xy = new double[2*order_num];
    double* xy1 = new double[2*order_num];
    double* xy2 = new double[2*order_num];
    double area1, area2;
    double* w = new double[order_num];
    // Compute untranslated rule
    dunavant_rule(rule, order_num, xy, w);
    QuadPoints ret;
    // Translate twice to break up the trapezoidal region
    // into two triangles
    // T1 = (t[i - 1], 0), (t[i - 1], t[i - 1]), (t[i], t[i])
    // T2 = (t[i - 1], 0), (t[i], 0), (t[i], t[i])
    const std::vector<double> t = eta->hidden_states;
    for (int i = 1; i < t.size(); ++i)
    {
        ret.emplace_back();
        double ti = std::min(100.0, t[i]);
        double node_xy1[] = {t[i - 1], 0, ti, 0, ti, ti};
        double node_xy2[] = {t[i - 1], 0, t[i - 1], t[i - 1], ti, ti};
        area1 = triangle_area(node_xy1);
        area2 = triangle_area(node_xy2);
        reference_to_physical_t3(node_xy1, order_num, xy, xy1);
        reference_to_physical_t3(node_xy2, order_num, xy, xy2);
        for (int order = 0; order < order_num; ++order)
        {
            ret.back().emplace_back(quad_point{xy1[0 + order * 2], xy1[1 + order + 2], w[order] * area1});
            if (area2 > 0)
                ret.back().emplace_back(quad_point{xy2[0 + order * 2], xy2[1 + order + 2], w[order] * area2});
        }
    }
    delete[] xy;
    delete[] xy1;
    delete[] xy2;
    delete[] w;
    return ret;
}

*/

/*
def i1(i, j):
    def f(u, h):
        a = -2 * (R(h) - R(u)) - R(max(h, t[j - 1]))
        b = -2 * (R(h) - R(u)) - R(max(h, t[j]))
        if a == b:
            return 0.
        return eta(h) / h * (np.exp(a) - np.exp(b))
    return scipy.integrate.dblquad(f, t[i - 1], t[i], lambda h: 0, lambda h: h)[0]

def i2(i, j):
    def f(u, h):
        a = 2 * R(u) - R(h) - 2 * R(max(u, min(t[j - 1], h)))
        b = 2 * R(u) - R(h) - 2 * R(max(u, min(t[j], h)))
        if a == b:
            return 0.
        return eta(h) / h * (np.exp(a) - np.exp(b))
    return scipy.integrate.dblquad(f, t[i - 1], t[i], lambda h: 0, lambda h: h)[0]

def i3(i, j):
    if i != j:
        return 0.
    def f(u, h):
        return eta(h) * np.exp(-R(h)) * -np.expm1(-2 * (R(h) - R(u))) / h
    return scipy.integrate.dblquad(f, t[i - 1], t[i], lambda h: 0, lambda h: h)[0]
    
def T(i, j):
    ret = i1(i, j) + 0.5 * i2(i, j) + 0.5 * i3(i, j)
    ret /= np.exp(-R(t[i - 1])) - np.exp(-R(t[i]))
    return ret

def P_no_recomb(i, rho):
    i1 = scipy.integrate.quad(lambda t: eta(t) * np.exp(-(R(t) + rho * t)), t[i - 1], t[i])[0]
    i1 /= np.exp(-R(t[i - 1])) - np.exp(-R(t[i]))
    return i1

*/

template <typename T>
Transition<T>::Transition(const PiecewiseExponentialRateFunction<T> &eta, double rho) :
    eta(&eta), M(eta.hidden_states.size()), Phi(M - 1, M - 1), rho(rho)
{
    Phi.setZero();
    compute();
}

template <typename T>
void Transition<T>::compute(void)
{
    for (int i = 1; i < M; ++i)
    {
        T pnr = P_no_recomb(i, rho);
        for (int j = 1; j < M; ++j)
        {
            Phi(i - 1, j - 1) = (1. - pnr) * trans(i, j);
            if (i == j)
                Phi(i - 1, j - 1) += pnr;
        }
    }
}

template <typename T>
Matrix<T>& Transition<T>::matrix(void) { return Phi; }

template class Transition<double>;
template class Transition<adouble>;

int main(int argc, char** argv)
{
    std::vector<std::vector<double> > params = {
        {0.5, 1.0},
        {0.5, 1.0},
        {0.1, 0.1}
    };
    std::vector<double> hs = {0.0, 1.0, 2.0, 15.0};
    std::vector<std::pair<int, int> > deriv = { {0,0} };
    PiecewiseExponentialRateFunction<adouble> eta(params, deriv, hs);
    Transition<adouble> T(eta, 4 * 1e4 * 1e-9);
    std::cout << T.matrix().template cast<double>() << std::endl;
}

