#ifndef SIMPSONS_H
#define SIMPSONS_H

template <typename T, typename U, typename V>
T adaptiveSimpsonsAux(const std::function<T(const V, U*)> &f, U* helper,
        V a, V b, double eps, T S, T fa, T fb, T fc, int bottom)
{
    V c = (a + b)/2, h = b - a;
    V d = (a + c)/2, e = (c + b)/2;
    T fd = f(d, helper), fe = f(e, helper);
    T Sleft = (h/12)*(fa + 4*fd + fc);
    T Sright = (h/12)*(fc + 4*fe + fb);
    T S2 = Sleft + Sright;
    T diff = S2 - S;
    if (bottom <= 0 || myabs(diff) <= 15*eps)
        return S2 + diff/15;
    return adaptiveSimpsonsAux(f, helper, a, c, eps/2, Sleft, fa, fc, fd, bottom-1) +
        adaptiveSimpsonsAux(f, helper, c, b, eps/2, Sright, fc, fb, fe, bottom-1);
}

template <typename T, typename U, typename V> 
T adaptiveSimpsons(const std::function<T(const V, U*)> &f, U* helper,
        V a, V b, double eps, int maxDepth)
{
    V c = (a + b)/2, h = b - a;
    T fa = f(a, helper), fb = f(b, helper), fc = f(c, helper);
    T S = (h/6)*(fa + 4*fc + fb);
    return adaptiveSimpsonsAux<T>(f, helper, a, b, eps, S, fa, fb, fc, maxDepth);
}

#endif
