/**
 * Copyright (c) 2018 by Robert Istvan Busa-Fekete
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 *
 * Based on liblinear API.
 */

#include <liblinear/linear.h>
#include <cstdlib>
#include <math.h>
#include <iostream>
#include "online_training.h"


model* train_online(const problem *prob, const online_parameter *param)
{
    int i,j,t;
    int l = prob->l;
    int n = prob->n;
    int w_size = prob->n;
    const int nr_class = 2;

    //allocate model
    model *model_ = Malloc(model,1);
    model_->w=Malloc(double, w_size);
    model_->nr_class = 2;

    if(prob->bias>=0)
        model_->nr_feature=n-1;
    else
        model_->nr_feature=n;
    //model_->param = *param;
    model_->bias = prob->bias;

    if(param->init_sol != NULL)
        for(i=0;i<w_size;i++)
            model_->w[i] = param->init_sol[i];
    else
        for(i=0;i<w_size;i++)
            model_->w[i] = 0;

    model_->label = Malloc(int,nr_class);
    for(i=0;i<nr_class;i++)
        model_->label[i] = i;

    model_->label[0]=1;
    model_->label[1]=0;

    feature_node **x=prob->x;
    double *y=prob->y;
    double *weight = param->weight;
    double eta = param->eta;

    bool deb = false;
    if (deb) {
        std::cerr << "\n";
        std::cerr << "Number of example: " << l << "\n";
    }
    for(t=0; t< param->iter; t++){
        for(i=0; i<l; i++){
            feature_node * const xi=x[i];
            double pred = sparse_operator::dot( model_->w, xi);
//            if (deb)
//                std::cerr << i << ". " << pred << "\n";
            double label = (y[i]>0.5) ? 1.0 : -1.0;
            double importance = (y[i]>0.5) ? param->weight[1] : param->weight[0];

            double negativeGrad = label / (1.0 + exp(label * pred)) * importance;

            if (abs(negativeGrad) > 1e-8) {
                double a = eta * sqrt(1.0 / (double)(t+1));
                sparse_operator::axpy(a*negativeGrad, xi, model_->w);
            }
        }
        if (deb) {
            double s2 = 0.0;
            for (j = 0; j < w_size; j++)
                s2 += (model_->w[j] * model_->w[j]);
            s2 = sqrt(s2);
            std::cerr << "-->  L2 " << t << ". " << s2 << "\n";

        //        for(j=0; j<w_size; j++)
        //            model_->w[j] /= s2;
        //        }
        }
    }

//    double s2 = 0.0;
//    for (j = 0; j < w_size; j++)
//        s2 += (model_->w[j] * model_->w[j]);
//    s2 = sqrt(s2);
//    if (isnan(s2) || isinf(s2))
//        std::cerr << "-->  L2 " << t << ". " << s2 << "\n";

    return model_;
}
