#!/usr/bin/env python

"""
This function is take from here: https://github.com/stur86/tvregdiff/blob/master/tvregdiff.py
"""

"""
Python function to estimate derivatives from noisy data based on
Rick Chartrand's Total Variation Regularized Numerical 
Differentiation (TVDiff) algorithm.
Example:
>>> u = TVRegDiff(data, iter, alph, u0, scale, ep, dx,  
...               plotflag, diagflag)    
Rick Chartrand (rickc@lanl.gov), Apr. 10, 2011
Please cite Rick Chartrand, "Numerical differentiation of noisy,
nonsmooth data," ISRN Applied Mathematics, Vol. 2011, Article ID 164564,
2011.
Copyright notice:
Copyright 2010. Los Alamos National Security, LLC. This material
was produced under U.S. Government contract DE-AC52-06NA25396 for
Los Alamos National Laboratory, which is operated by Los Alamos
National Security, LLC, for the U.S. Department of Energy. The
Government is granted for, itself and others acting on its
behalf, a paid-up, nonexclusive, irrevocable worldwide license in
this material to reproduce, prepare derivative works, and perform
publicly and display publicly. Beginning five (5) years after
(March 31, 2011) permission to assert copyright was obtained,
subject to additional five-year worldwide renewals, the
Government is granted for itself and others acting on its behalf
a paid-up, nonexclusive, irrevocable worldwide license in this
material to reproduce, prepare derivative works, distribute
copies to the public, perform publicly and display publicly, and
to permit others to do so. NEITHER THE UNITED STATES NOR THE
UNITED STATES DEPARTMENT OF ENERGY, NOR LOS ALAMOS NATIONAL
SECURITY, LLC, NOR ANY OF THEIR EMPLOYEES, MAKES ANY WARRANTY,
EXPRESS OR IMPLIED, OR ASSUMES ANY LEGAL LIABILITY OR
RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF
ANY INFORMATION, APPARATUS, PRODUCT, OR PROCESS DISCLOSED, OR
REPRESENTS THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED
RIGHTS.
BSD License notice:
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
     Redistributions of source code must retain the above
     copyright notice, this list of conditions and the following
     disclaimer.
     Redistributions in binary form must reproduce the above
     copyright notice, this list of conditions and the following
     disclaimer in the documentation and/or other materials
     provided with the distribution.
     Neither the name of Los Alamos National Security nor the names of its
     contributors may be used to endorse or promote products
     derived from this software without specific prior written
     permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
#########################################################
#                                                       #
# Python translation by Simone Sturniolo                #
# Rutherford Appleton Laboratory, STFC, UK (2014)       #
# simonesturniolo@gmail.com                             #
#                                                       #
#########################################################
"""

import sys

try:
    import numpy as np
    import scipy as sp
    from scipy import sparse
    from scipy.sparse import linalg as splin
except ImportError:
    sys.exit("Numpy and Scipy must be installed for TVRegDiag to work - "
             "aborting")

_has_matplotlib = True

try:
    import matplotlib.pyplot as plt
except ImportError:
    _has_matplotlib = False
    print("Matplotlib is not installed - plotting functionality disabled")

# Utility function.


def chop(v):
    return v[1:]


def TVRegDiff(data, itern, alph, u0=None, scale='small', ep=1e-6, dx=None,
              plotflag=_has_matplotlib, diagflag=1):

    # code starts here
    # Make sure we have a column vector
    data = np.array(data)
    if (len(data.shape) != 1):
        print("Error - data is not a column vector")
        return
    # Get the data size.
    n = len(data)

    # Default checking. (u0 is done separately within each method.)
    if dx is None:
        dx = 1.0 / n

    # Different methods for small- and large-scale problems.
    if (scale.lower() == 'small'):

        # Construct differentiation matrix.
        c = np.ones(n + 1) / dx
        D = sparse.spdiags([-c, c], [0, 1], n, n + 1)

        DT = D.transpose()

        # Construct antidifferentiation operator and its adjoint.
        def A(x): return chop(np.cumsum(x) - 0.5 * (x + x[0])) * dx

        def AT(w): return (sum(w) * np.ones(n + 1) -
                           np.transpose(np.concatenate(([sum(w) / 2.0],
                                                        np.cumsum(w) -
                                                        w / 2.0)))) * dx

        # Default initialization is naive derivative

        if u0 is None:
            u0 = np.concatenate(([0], np.diff(data), [0]))

        u = u0
        # Since Au( 0 ) = 0, we need to adjust.
        ofst = data[0]
        # Precompute.
        ATb = AT(ofst - data)        # input: size n

        # Main loop.
        for ii in range(1, itern+1):
            # Diagonal matrix of weights, for linearizing E-L equation.
            Q = sparse.spdiags(1. / (np.sqrt((D * u)**2 + ep)), 0, n, n)
            # Linearized diffusion matrix, also approximation of Hessian.
            L = dx * DT * Q * D

            # Gradient of functional.
            g = AT(A(u)) + ATb + alph * L * u

            # Prepare to solve linear equation.
            tol = 1e-4
            maxit = 100
            # Simple preconditioner.
            P = alph * sparse.spdiags(L.diagonal() + 1, 0, n + 1, n + 1)

            def linop(v): return (alph * L * v + AT(A(v)))
            linop = splin.LinearOperator((n + 1, n + 1), linop)

            if diagflag:
                [s, info_i] = sparse.linalg.cg(
                    linop, g, x0=None, tol=tol, maxiter=maxit, callback=None,
                    M=P)
                print('iteration {0:4d}: relative change = {1:.3e}, '
                      'gradient norm = {2:.3e}\n'.format(ii,
                                                         np.linalg.norm(
                                                             s[0]) /
                                                         np.linalg.norm(u),
                                                         np.linalg.norm(g)))
                if (info_i > 0):
                    print("WARNING - convergence to tolerance not achieved!")
                elif (info_i < 0):
                    print("WARNING - illegal input or breakdown")
            else:
                [s, info_i] = sparse.linalg.cg(
                    linop, g, x0=None, tol=tol, maxiter=maxit, callback=None,
                    M=P)
            # Update solution.
            u = u - s
            # Display plot.
            if plotflag:
                plt.plot(u)
                plt.show()

    elif (scale.lower() == 'large'):

        # Construct antidifferentiation operator and its adjoint.
        def A(v): return np.cumsum(v)

        def AT(w): return (sum(w) * np.ones(len(w)) -
                           np.transpose(np.concatenate(([0.0],
                                                        np.cumsum(w[:-1])))))
        # Construct differentiation matrix.
        c = np.ones(n)
        D = sparse.spdiags([-c, c], [0, 1], n, n) / dx
        mask = np.ones((n, n))
        mask[-1, -1] = 0.0
        D = sparse.dia_matrix(D.multiply(mask))
        DT = D.transpose()
        # Since Au( 0 ) = 0, we need to adjust.
        data = data - data[0]
        # Default initialization is naive derivative.
        if u0 is None:
            u0 = np.concatenate(([0], np.diff(data)))
        u = u0
        # Precompute.
        ATd = AT(data)

        # Main loop.
        for ii in range(1, itern + 1):
            # Diagonal matrix of weights, for linearizing E-L equation.
            Q = sparse.spdiags(1. / np.sqrt((D*u)**2.0 + ep), 0, n, n)
            # Linearized diffusion matrix, also approximation of Hessian.
            L = DT*Q*D
            # Gradient of functional.
            g = AT(A(u)) - ATd
            g = g + alph * L * u
            # Build preconditioner.
            c = np.cumsum(range(n, 0, -1))
            B = alph * L + sparse.spdiags(c[::-1], 0, n, n)
            # droptol = 1.0e-2
            R = sparse.dia_matrix(np.linalg.cholesky(B.todense()))
            # Prepare to solve linear equation.
            tol = 1.0e-4
            maxit = 100

            def linop(v): return (alph * L * v + AT(A(v)))
            linop = splin.LinearOperator((n, n), linop)

            if diagflag:
                [s, info_i] = sparse.linalg.cg(
                    linop, -g, x0=None, tol=tol, maxiter=maxit, callback=None,
                    M=np.dot(R.transpose(), R))
                print('iteration {0:4d}: relative change = {1:.3e}, '
                      'gradient norm = {2:.3e}\n'.format(ii,
                                                         np.linalg.norm(s[0]) /
                                                         np.linalg.norm(u),
                                                         np.linalg.norm(g)))
                if (info_i > 0):
                    print("WARNING - convergence to tolerance not achieved!")
                elif (info_i < 0):
                    print("WARNING - illegal input or breakdown")

            else:
                [s, info_i] = sparse.linalg.cg(
                    linop, -g, x0=None, tol=tol, maxiter=maxit, callback=None,
                    M=np.dot(R.transpose(), R))
            # Update current solution
            u = u + s
            # Display plot.
            if plotflag:
                plt.plot(u/dx)
                plt.show()

        u = u/dx

    return u

# Small testing script


if __name__ == "__main__":

    dx = 0.05
    x0 = np.arange(0, 2.0*np.pi, dx)

    testf = np.sin(x0)

    testf = testf + np.random.normal(0.0, 0.04, x0.shape)

    deriv_sm = TVRegDiff(testf, 1, 5e-2, dx=dx,
                         ep=1e-1, scale='small', plotflag=0)
    deriv_lrg = TVRegDiff(testf, 100, 1e-1, dx=dx,
                          ep=1e-2, scale='large', plotflag=0)

    if (_has_matplotlib):
        plt.plot(np.cos(x0), label='Analytical', c=(0,0,0))
        plt.plot(np.gradient(testf, dx), label='numpy.gradient')
        plt.plot(deriv_sm, label='TVRegDiff (small)')
        plt.plot(deriv_lrg, label='TVRegDiff (large)')
        plt.legend()
        plt.show()

