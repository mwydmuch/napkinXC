#ifndef _LIBLINEAR_H
#define _LIBLINEAR_H

#define LIBLINEAR_VERSION 230
#include <iostream>

struct feature_node{
    int index;
    float value;
};

#ifdef __cplusplus
extern "C" {
#endif

extern int liblinear_version;

struct problem
{
	int l, n;
	float *y;
	struct feature_node **x;
	float bias;            /* < 0 if no bias term */
	float *W;              /* instance weight */
};

enum { L2R_LR, L2R_L2LOSS_SVC_DUAL, L2R_L2LOSS_SVC, L2R_L1LOSS_SVC_DUAL, MCSVM_CS, L1R_L2LOSS_SVC, L1R_LR, L2R_LR_DUAL, L2R_L2LOSS_SVR = 11, L2R_L2LOSS_SVR_DUAL, L2R_L1LOSS_SVR_DUAL }; /* solver_type */

struct parameter
{
	int solver_type;

	/* these are for training only */
	float eps;	        /* stopping criteria */
	float C;
	int nr_weight;
	int *weight_label;
	float* weight;
	float p;
	float *init_sol;
	int max_iter;
};

struct model
{
	struct parameter param;
	int nr_class;		/* number of classes */
	int nr_feature;
	float *w;
	int *label;		/* label of each class */
	float bias;
};

struct model* train_liblinear(const struct problem *prob, const struct parameter *param);
void cross_validation(const struct problem *prob, const struct parameter *param, int nr_fold, float *target);
void find_parameters(const struct problem *prob, const struct parameter *param, int nr_fold, float start_C, float start_p, float *best_C, float *best_p, float *best_score);

float predict_values(const struct model *model_, const struct feature_node *x, float* dec_values);
float predict(const struct model *model_, const struct feature_node *x);
float predict_probability(const struct model *model_, const struct feature_node *x, float* prob_estimates);

int save_model(const char *model_file_name, const struct model *model_);
struct model *load_model(const char *model_file_name);

int get_nr_feature(const struct model *model_);
int get_nr_class(const struct model *model_);
void get_labels(const struct model *model_, int* label);
float get_decfun_coef(const struct model *model_, int feat_idx, int label_idx);
float get_decfun_bias(const struct model *model_, int label_idx);

void free_model_content(struct model *model_ptr);
void free_and_destroy_model(struct model **model_ptr_ptr);
void destroy_param(struct parameter *param);

const char *check_parameter(const struct problem *prob, const struct parameter *param);
int check_probability_model(const struct model *model);
int check_regression_model(const struct model *model);
void set_print_string_function(void (*print_func) (const char*));

#ifdef __cplusplus
}
#endif

#endif /* _LIBLINEAR_H */

