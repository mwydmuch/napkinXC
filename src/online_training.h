//
// Created by Robert Busa-Fekete on 4/19/18.
//

#ifndef NAPKINXML_ONLINE_TRAINING_H
#define NAPKINXML_ONLINE_TRAINING_H

#endif //NAPKINXML_ONLINE_TRAINING_H

struct online_parameter {
    int iter;
    double eta;
    int nr_weight;
    int *weight_label;
    double* weight;
    double p;
    double *init_sol;
};


struct model* train_online(const struct problem *prob, const struct online_parameter *param);