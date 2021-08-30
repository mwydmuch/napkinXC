#ifndef _TRON_H
#define _TRON_H

class function
{
public:
	virtual float fun(float *w) = 0 ;
	virtual void grad(float *w, float *g) = 0 ;
	virtual void Hv(float *s, float *Hs) = 0 ;

	virtual int get_nr_variable(void) = 0 ;
	virtual void get_diag_preconditioner(float *M) = 0 ;
	virtual ~function(void){}
};

class TRON
{
public:
	TRON(const function *fun_obj, float eps = 0.1, float eps_cg = 0.1, int max_iter = 1000);
	~TRON();

	void tron(float *w);
	void set_print_string(void (*i_print) (const char *buf));

private:
	int trpcg(float delta, float *g, float *M, float *s, float *r, bool *reach_boundary);
	float norm_inf(int n, float *x);

	float eps;
	float eps_cg;
	int max_iter;
	function *fun_obj;
	void info(const char *fmt,...);
	void (*tron_print_string)(const char *buf);
};
#endif
