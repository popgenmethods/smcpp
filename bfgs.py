from __future__ import division
import numpy as np

def bfgs(f, fprime, x0, epsi = 10e-8, tol=10e-6, sigma=10**-1, beta=10):
    def LineSearch(g, x, s,sigma=10**-1, beta=10, convergval=0.00001):
        # QFind returns 1 or the proper value (based on the current slope) of the line
        # based on basic rise over run of the distance of the current function with
        # new vs old value
        def QFind(alpha):
            if np.abs(alpha) < convergval:
                return 1
            return (f(x + alpha * s) - f(x))/(alpha * np.dot(g,s))

        alpha = 0.001

        # Double alpha until big enough
        while QFind(alpha) >= sigma:
            alpha *= 2.0

        # back track
        while QFind(alpha) < sigma:
            alphap = alpha / ( 2.0* ( 1- QFind(alpha))) 
            alpha = max(1.0/beta * alpha, alphap)
        return alpha

    # Startup
    x = x0
    xold = np.inf
    N = x.shape[0]
    H = 1.0 * np.eye(N)
    counter = 1
    alpha = 1
    g = fprime(x)
    while np.linalg.norm(g) > epsi and np.linalg.norm(xold - x) > tol:
        s = -np.dot(H,g)
        # Repeating the linesearch
        alpha = LineSearch(g,x,s)
        x += alpha * s
        gold = g
        g = fprime(x)
        y = (g - gold)/alpha
        dotsy = np.dot(s,y)
        if dotsy > 0:
            # Update H using estimation technique
            z = np.dot(H,y)
            H += np.outer(s,s)*(np.dot(s,y) + np.dot(y, z))/dotsy**2 - \
                    (np.outer(z,s) + np.outer(s, z))/dotsy
        counter+=1

    return (x , counter)
