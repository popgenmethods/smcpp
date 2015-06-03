/*
	Numerical Integration by Gauss-Legendre Quadrature Formulas of high orders.
	High-precision abscissas and weights are used.

	Project homepage: http://www.holoborodko.com/pavel/?page_id=679
	Contact e-mail:   pavel@holoborodko.com

	Copyright (c)2007-2010 Pavel Holoborodko
	All rights reserved.

	Redistribution and use in source and binary forms, with or without
	modification, are permitted provided that the following conditions
	are met:
	
	1. Redistributions of source code must retain the above copyright
	notice, this list of conditions and the following disclaimer.
	
	2. Redistributions in binary form must reproduce the above copyright
	notice, this list of conditions and the following disclaimer in the
	documentation and/or other materials provided with the distribution.
	
	3. Redistributions of any form whatsoever must retain the following
	acknowledgment:
	"
         This product includes software developed by Pavel Holoborodko
         Web: http://www.holoborodko.com/pavel/
         e-mail: pavel@holoborodko.com
	
	"

	4. This software cannot be, by any means, used for any commercial 
	purpose without the prior permission of the copyright holder.
	
	Any of the above conditions can be waived if you get permission from 
	the copyright holder. 

	THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
	ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
	IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
	ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
	FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
	DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
	OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
	HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
	LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
	OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
	SUCH DAMAGE.

	Contributors
	Konstantin Holoborodko - Optimization of Legendre polynomial computing.
*/

#ifndef __GAUSS_LEGENDRE_H__
#define __GAUSS_LEGENDRE_H__

	/* Numerical computation of int(f(x),x=a..b) by Gauss-Legendre n-th order high precision quadrature 
		[in]n     - quadrature order
		[in]f     - integrand
		[in]data  - pointer on user-defined data which will 
				    be passed to f every time it called (as second parameter).
		[in][a,b] - interval of integration
	   
		return:
	         -computed integral value or -1.0 if n order quadrature is not supported
	*/
    template <typename T>
    T gauss_legendre(int n, T (*f)(T, void*), void* data, T a, T b);

	/* 2D Numerical computation of int(f(x,y),x=a..b,y=c..d) by Gauss-Legendre n-th order high precision quadrature 
		[in]n     - quadrature order
		[in]f     - integrand
		[in]data  - pointer on user-defined data which will 
					be passed to f every time it called (as third parameter).
		[in][a,b]x[c,d] - interval of integration

	return:
			-computed integral value or -1.0 if n order quadrature is not supported
	*/
	double gauss_legendre_2D_cube(int n, double (*f)(double,double,void*), void* data, double a, double b, double c, double d);

	/* Computing of abscissas and weights for Gauss-Legendre quadrature for any(reasonable) order n
		[in] n   - order of quadrature
		[in] eps - required precision (must be eps>=macheps(double), usually eps = 1e-10 is ok)
		[out]x   - abscisass, size = (n+1)>>1
		[out]w   - weights, size = (n+1)>>1 
	*/
	void gauss_legendre_tbl(int n, double* x, double* w, double eps);

#endif /* __GAUSS_LEGENDRE_H__ */


