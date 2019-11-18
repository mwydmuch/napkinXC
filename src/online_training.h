/**
 * Copyright (c) 2018 by Robert Istvan Busa-Fekete
 * All rights reserved.
 *
 * Based on liblinear API.
 */

#pragma once

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
