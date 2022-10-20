#include "tron.h"
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif

#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif

#ifdef __cplusplus
extern "C" {
#endif

extern float nrm2_(int *, float *, int *);
extern float dot_(int *, float *, int *, float *, int *);
extern int axpy_(int *, float *, float *, int *, float *, int *);
extern int scal_(int *, float *, float *, int *);

#ifdef __cplusplus
}
#endif

static void default_print(const char *buf)
{
	fputs(buf,stdout);
	fflush(stdout);
}

static float uTMv(int n, float *u, float *M, float *v)
{
	const int m = n-4;
	float res = 0;
	int i;
	for (i=0; i<m; i+=5)
		res += u[i]*M[i]*v[i]+u[i+1]*M[i+1]*v[i+1]+u[i+2]*M[i+2]*v[i+2]+
			u[i+3]*M[i+3]*v[i+3]+u[i+4]*M[i+4]*v[i+4];
	for (; i<n; i++)
		res += u[i]*M[i]*v[i];
	return res;
}

void TRON::info(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*tron_print_string)(buf);
}

TRON::TRON(const function *fun_obj, float eps, float eps_cg, int max_iter)
{
	this->fun_obj=const_cast<function *>(fun_obj);
	this->eps=eps;
	this->eps_cg=eps_cg;
	this->max_iter=max_iter;
	tron_print_string = default_print;
}

TRON::~TRON()
{
}

void TRON::tron(float *w)
{
	// Parameters for updating the iterates.
	float eta0 = 1e-4, eta1 = 0.25, eta2 = 0.75;

	// Parameters for updating the trust region size delta.
	float sigma1 = 0.25, sigma2 = 0.5, sigma3 = 4;

	int n = fun_obj->get_nr_variable();
	int i, cg_iter;
	float delta=0, sMnorm, one=1.0;
	float alpha, f, fnew, prered, actred, gs;
	int search = 1, iter = 1, inc = 1;
	float *s = new float[n];
	float *r = new float[n];
	float *g = new float[n];

	const float alpha_pcg = 0.01;
	float *M = new float[n];

	// calculate gradient norm at w=0 for stopping condition.
	float *w0 = new float[n];
	for (i=0; i<n; i++)
		w0[i] = 0;
	fun_obj->fun(w0);
	fun_obj->grad(w0, g);
	float gnorm0 = nrm2_(&n, g, &inc);
	delete [] w0;

	f = fun_obj->fun(w);
	fun_obj->grad(w, g);
	float gnorm = nrm2_(&n, g, &inc);

	if (gnorm <= eps*gnorm0)
		search = 0;

	fun_obj->get_diag_preconditioner(M);
	for(i=0; i<n; i++)
		M[i] = (1-alpha_pcg) + alpha_pcg*M[i];
	delta = sqrt(uTMv(n, g, M, g));

	float *w_new = new float[n];
	bool reach_boundary;
	bool delta_adjusted = false;
	while (iter <= max_iter && search)
	{
		cg_iter = trpcg(delta, g, M, s, r, &reach_boundary);

		memcpy(w_new, w, sizeof(float)*n);
		axpy_(&n, &one, s, &inc, w_new, &inc);

		gs = dot_(&n, g, &inc, s, &inc);
		prered = -0.5*(gs-dot_(&n, s, &inc, r, &inc));
		fnew = fun_obj->fun(w_new);

		// Compute the actual reduction.
		actred = f - fnew;

		// On the first iteration, adjust the initial step bound.
		sMnorm = sqrt(uTMv(n, s, M, s));
		if (iter == 1 && !delta_adjusted)
		{
			delta = min(delta, sMnorm);
			delta_adjusted = true;
		}

		// Compute prediction alpha*sMnorm of the step.
		if (fnew - f - gs <= 0)
			alpha = sigma3;
		else
			alpha = max(sigma1, -0.5f*(gs/(fnew - f - gs)));

		// Update the trust region bound according to the ratio of actual to predicted reduction.
		if (actred < eta0*prered)
			delta = min(alpha*sMnorm, sigma2*delta);
		else if (actred < eta1*prered)
			delta = max(sigma1*delta, min(alpha*sMnorm, sigma2*delta));
		else if (actred < eta2*prered)
			delta = max(sigma1*delta, min(alpha*sMnorm, sigma3*delta));
		else
		{
			if (reach_boundary)
				delta = sigma3*delta;
			else
				delta = max(delta, min(alpha*sMnorm, sigma3*delta));
		}

		//info("iter %2d act %5.3e pre %5.3e delta %5.3e f %5.3e |g| %5.3e CG %3d\n", iter, actred, prered, delta, f, gnorm, cg_iter);

		if (actred > eta0*prered)
		{
			iter++;
			memcpy(w, w_new, sizeof(float)*n);
			f = fnew;
			fun_obj->grad(w, g);
			fun_obj->get_diag_preconditioner(M);
			for(i=0; i<n; i++)
				M[i] = (1-alpha_pcg) + alpha_pcg*M[i];

			gnorm = nrm2_(&n, g, &inc);
			if (gnorm <= eps*gnorm0)
				break;
		}
		if (f < -1.0e+32)
		{
			//info("WARNING: f < -1.0e+32\n");
			break;
		}
		if (prered <= 0)
		{
			//info("WARNING: prered <= 0\n");
			break;
		}
		if (fabs(actred) <= 1.0e-12*fabs(f) &&
		    fabs(prered) <= 1.0e-12*fabs(f))
		{
			//info("WARNING: actred and prered too small\n");
			break;
		}
	}

	delete[] g;
	delete[] r;
	delete[] w_new;
	delete[] s;
	delete[] M;
}

int TRON::trpcg(float delta, float *g, float *M, float *s, float *r, bool *reach_boundary)
{
	int i, inc = 1;
	int n = fun_obj->get_nr_variable();
	float one = 1;
	float *d = new float[n];
	float *Hd = new float[n];
	float zTr, znewTrnew, alpha, beta, cgtol;
	float *z = new float[n];

	*reach_boundary = false;
	for (i=0; i<n; i++)
	{
		s[i] = 0;
		r[i] = -g[i];
		z[i] = r[i] / M[i];
		d[i] = z[i];
	}

	zTr = dot_(&n, z, &inc, r, &inc);
	cgtol = eps_cg*sqrt(zTr);
	int cg_iter = 0;
	int max_cg_iter = max(n, 5);

	while (cg_iter < max_cg_iter)
	{
		if (sqrt(zTr) <= cgtol)
			break;
		cg_iter++;
		fun_obj->Hv(d, Hd);

		alpha = zTr/dot_(&n, d, &inc, Hd, &inc);
		axpy_(&n, &alpha, d, &inc, s, &inc);

		float sMnorm = sqrt(uTMv(n, s, M, s));
		if (sMnorm > delta)
		{
			//info("cg reaches trust region boundary\n");
			*reach_boundary = true;
			alpha = -alpha;
			axpy_(&n, &alpha, d, &inc, s, &inc);

			float sTMd = uTMv(n, s, M, d);
			float sTMs = uTMv(n, s, M, s);
			float dTMd = uTMv(n, d, M, d);
			float dsq = delta*delta;
			float rad = sqrt(sTMd*sTMd + dTMd*(dsq-sTMs));
			if (sTMd >= 0)
				alpha = (dsq - sTMs)/(sTMd + rad);
			else
				alpha = (rad - sTMd)/dTMd;
			axpy_(&n, &alpha, d, &inc, s, &inc);
			alpha = -alpha;
			axpy_(&n, &alpha, Hd, &inc, r, &inc);
			break;
		}
		alpha = -alpha;
		axpy_(&n, &alpha, Hd, &inc, r, &inc);

		for (i=0; i<n; i++)
			z[i] = r[i] / M[i];
		znewTrnew = dot_(&n, z, &inc, r, &inc);
		beta = znewTrnew/zTr;
		scal_(&n, &beta, d, &inc);
		axpy_(&n, &one, z, &inc, d, &inc);
		zTr = znewTrnew;
	}

	if (cg_iter == max_cg_iter)
		//info("WARNING: reaching maximal number of CG steps\n");

	delete[] d;
	delete[] Hd;
	delete[] z;

	return(cg_iter);
}

float TRON::norm_inf(int n, float *x)
{
	float dmax = fabs(x[0]);
	for (int i=1; i<n; i++)
		if (fabs(x[i]) >= dmax)
			dmax = fabs(x[i]);
	return(dmax);
}

void TRON::set_print_string(void (*print_string) (const char *buf))
{
	tron_print_string = print_string;
}
