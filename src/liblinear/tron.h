#ifndef _TRON_H
#define _TRON_H

#define REAL double

class function
{
public:
	virtual REAL fun(REAL *w) = 0 ;
	virtual void grad(REAL *w, REAL *g) = 0 ;
	virtual void Hv(REAL *s, REAL *Hs) = 0 ;

	virtual int get_nr_variable(void) = 0 ;
	virtual void get_diag_preconditioner(REAL *M) = 0 ;
	virtual ~function(void){}
};

class TRON
{
public:
	TRON(const function *fun_obj, REAL eps = 0.1, REAL eps_cg = 0.1, int max_iter = 1000);
	~TRON();

	void tron(REAL *w);
	void set_print_string(void (*i_print) (const char *buf));

private:
	int trpcg(REAL delta, REAL *g, REAL *M, REAL *s, REAL *r, bool *reach_boundary);
	REAL norm_inf(int n, REAL *x);

	REAL eps;
	REAL eps_cg;
	int max_iter;
	function *fun_obj;
	void info(const char *fmt,...);
	void (*tron_print_string)(const char *buf);
};
#endif
