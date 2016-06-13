from __future__ import absolute_import, division, print_function
import numpy as np
import sys
import logging

logger = logging.getLogger(__name__)

# NOT IMPLEMENTED:
# Cubic line search is not implemented (Currently only halving)

# Options:
# verbose: level of verbosity (0: no output, 1: final, 2: iter (default), 3:debug)
# optTol: tolerance used to check for optimality
# progTol: tolerance used to check for lack of progress
# maxIter: maximum number of calls to funObj
# suffDec: sufficient decrease parameter in Armijo condition
# curvilinear: backtrack along projection arc
# memory: number of steps to look back in non-monotone Armijo condition
# bbType: type of Barzilai-Borwein step
# interp: 0=none, 2=cubic (for the most part.. see below)
# numDiff: compute derivatives numerically (0: use user-supplied derivatives (default), 1: use finite differences)

# V0.2 Feb 15th 2014
# Python code by: Tomer Levinboim (first.last at usc.edu)
# Original matlab code by Mark Schmidt:
# http://www.di.ens.fr/~mschmidt/Software/minConf.html
class SPGOptions():
        pass


def projectBound(x, LB, UB=None):
    x[x < LB] = LB[x < LB]
    if UB is not None:
        x[x > UB] = UB[x > UB]
    return x


default_options = SPGOptions()
default_options.maxIter = 500
default_options.verbose = 2
default_options.suffDec = 1e-4
default_options.progTol = 1e-9
default_options.optTol = .01
default_options.curvilinear = False
default_options.memory = 10
default_options.useSpectral = False
default_options.bbType = 1
default_options.interp = 0  # cubic
default_options.numdiff = 0
default_options.testOpt = True


def assertVector(v):
    assert len(v.shape) == 1


def SPG(funObj, funProj, x, options=default_options):
    x = funProj(x)
    f, g = funObj(x)
    dx = np.zeros_like(x)
    i = 0  # iteration
    for i in range(5):
        gn = np.linalg.norm(g)
        t = 2. / gn / (i + 1)
        xp = funProj(x - t * g)
        fp, gp = funObj(xp)
        j = 0
        while fp > f - t * 1e-3 * gn**2:
            t *= 0.5
            xp = funProj(x - t * g)
            fp, gp = funObj(xp)
            j += 1
            logger.debug("line search: %g <> %g" % (fp, f - t * 1e-3 * gn**2))
        xp = funProj(xp + 0.5 * dx)
	fp, gp = funObj(xp)
        dx = xp - x
        df = fp - f
        logger.debug("i:%d df:%g |dx|:%g |g|:%g" % (i, df, np.linalg.norm(dx), np.linalg.norm(gp)))
        x, f, g = xp, fp, gp
        if np.abs(dx).max() < .01:
            break
    return x, f


def SPG2(funObj0, funProj, x, options=default_options):
    x = funProj(x)
    i = 1  # iteration

    funEvalMultiplier = 1
    if options.numdiff == 1:
        funObj = lambda x: auto_grad(x, funObj0, options)
        funEvalMultiplier = len(x)+1
    else:
        funObj = funObj0

    f, g = funObj(x)
    projects = 1
    funEvals = 1

    if options.verbose >= 2:
        if options.testOpt:
            print('%10s %10s %10s %15s %15s %15s' % ('Iteration', 'FunEvals', 'Projections', 'Step Length', 'Function Val', 'Opt Cond'))
        else:
            print('%10s %10s %10s %15s %15s' % ('Iteration', 'FunEvals', 'Projections', 'Step Length', 'Function Val'))

    while funEvals <= options.maxIter:
        if i == 1 or not options.useSpectral:
            alpha = 1**(-i)
        else:
            y = g - g_old
            s = x - x_old
            assertVector(y)
            assertVector(s)

            # type of BB step
            if options.bbType == 1:
                alpha = np.dot(s.T, s) / np.dot(s.T, y)
            else:
                alpha = np.dot(s.T, y) / np.dot(y.T, y)

            if alpha <= 1e-10 or alpha > 1e10:
                alpha = 1.0

        d = -alpha * g / np.linalg.norm(g)
        f_old = f
        x_old = x
        g_old = g

        if not options.curvilinear:
            d = funProj(x + d) - x
            projects += 1

        gtd = np.dot(g, d)

        if gtd > -options.progTol:
            logger.debug('Directional Derivative below progTol')
            break

        if i == 1:
            t = min([1, 1.0 / np.sum(np.absolute(g))])
        else:
            t = 1.0

        if options.memory == 1:
            funRef = f
        else:
            if i == 1:
                old_fvals = np.tile(-np.inf, (options.memory, 1))

            if i <= options.memory:
                old_fvals[i - 1] = f
            else:
                old_fvals = np.vstack([old_fvals[1:], f])

            funRef = np.max(old_fvals)

        if options.curvilinear:
            x_new = funProj(x + t * d)
            projects += 1
        else:
            x_new = x + t * d

        f_new, g_new = funObj(x_new)
        funEvals += 1
        lineSearchIters = 1
        while f_new > funRef + options.suffDec * np.dot(g.T, (x_new - x)) or not isLegal(f_new):
            temp = t
            # Halfing step size
            if options.interp == 0 or ~isLegal(f_new):
                logger.debug('Halving Step Size')
                t /= 2.0
            elif options.interp == 2 and isLegal(g_new):
                logger.debug('Cubic Backtracking')
                gtd_new = np.dot(g_new, d)
                t = polyinterp2(np.array([[0, f, gtd], [t, f_new, gtd_new]]))
            elif lineSearchIters < 2 or ~isLegal(f_prev):
                logger.debug('Quadratic Backtracking')
                t = polyinterp2(np.array([[0, f, gtd], [t, f_new, 1j]])).real
            else:
                # t = polyinterp([0 f gtd; t f_new sqrt(-1);t_prev f_prev sqrt(-1)]);
                # not implemented.
                # fallback on halving.
                t /= 2.0

            if t < temp * 1e-3:
                logger.debug('Interpolated value too small, Adjusting: ' + str(t))
                t = temp * 1e-3
            elif t > temp * 0.6:
                logger.debug('Interpolated value too large, Adjusting: ' + str(t))
                t = temp * 0.6
            # Check whether step has become too small
            if np.max(np.absolute(t * d)) < options.progTol or t == 0:
                logger.debug('Line Search failed')
                t = 0.0
                f_new = f
                g_new = g
                break

            # Evaluate New Point
            f_prev = f_new
            t_prev = temp

            if options.curvilinear:
                x_new = funProj(x + t * d)
                projects += 1
            else:
                x_new = x + t * d

            f_new, g_new = funObj(x_new)
            funEvals += 1
            lineSearchIters += 1

        # Take Step
        x = x_new
        f = f_new
        g = g_new

        if options.testOpt:
            optCond = np.max(np.absolute(funProj(x - g) - x))
            projects += 1

        # Output Log
        if options.verbose >= 2:
            if options.testOpt:
                print('{:10d} {:10d} {:10d} {:15.5e} {:15.5e} {:15.5e}'.format(i, funEvals * funEvalMultiplier,
                                                                               projects, t, f, optCond))
            else:
                print('{:10d} {:10d} {:10d} {:15.5e} {:15.5e}'.format(i, funEvals * funEvalMultiplier, projects, t, f))

        # Check optimality
        if options.testOpt:
            if optCond < options.optTol:
                logger.debug('First-Order Optimality Conditions Below optTol')
                break

        if np.max(np.absolute(t * d)) < options.progTol:
            logger.debug('Step size below progTol')
            break

        if np.absolute(f - f_old) < options.progTol:
            logger.debug('Function value changing by less than progTol')
            break

        if funEvals * funEvalMultiplier > options.maxIter:
            logger.debug('Function Evaluations exceeds maxIter')
            break

        i += 1

    return x, f


def isLegal(v):
    no_complex = np.imag(v).any().sum() == 0
    no_nan = np.isnan(v).sum() == 0
    no_inf = np.isinf(v).sum() == 0
    return no_complex and no_nan and no_inf


def polyinterp2(points):
     # Code for most common case:
     #   - cubic interpolation of 2 points w/ function and derivative values for both
     #   - no xminBound/xmaxBound
     # Solution in this case (where x2 is the farthest point):
     #    d1 = g1 + g2 - 3*(f1-f2)/(x1-x2);
     #    d2 = sqrt(d1^2 - g1*g2);
     #    minPos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
     #    t_new = min(max(minPos,x1),x2);
    minPos = np.argmin(points[:, 0])
    # minVal = points[minPos, 0]
    notMinPos = -minPos + 1
    d1 = points[minPos, 2] + points[notMinPos, 2] - 3*(points[minPos, 1]-points[notMinPos, 1])/(points[minPos, 0] - points[notMinPos, 0])
    d2 = np.sqrt(d1**2 - points[minPos, 2] * points[notMinPos,2])
    if np.isreal(d2):
        t = points[notMinPos, 0] - (points[notMinPos, 0] - points[minPos, 0])*((points[notMinPos, 2] + d2 - d1) / (points[notMinPos, 2] - points[minPos, 2] + 2*d2))
        minPos = min([max([t, points[minPos, 0]]), points[notMinPos, 0]])
    else:
        minPos = np.mean(points[:, 0])
    return minPos


def auto_grad(x, funObj, options):
    # notice the funObj should return a single value here - the objective (i.e., no gradient)
    p = len(x)
    f = funObj(x)
    if type(f) == type(()):
        f = f[0]

    mu = 2*np.sqrt(1e-12)*(1+np.linalg.norm(x))/np.linalg.norm(p)
    diff = np.zeros((p,))
    for j in range(p):
        e_j = np.zeros((p,))
        e_j[j] = 1
        # this is somewhat wrong, since we also need to project,
        # but practically (and locally) it doesn't seem to matter.
        v = funObj(x + mu*e_j)
        if type(v) == type(()):
            diff[j] = v[0]
        else:
            diff[j] = v

    g = (diff-f)/mu

    return f, g
