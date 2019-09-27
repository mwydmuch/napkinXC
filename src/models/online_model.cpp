/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include "online_model.h"


int onlineTrainThread(int threadId, Model* model, SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args,
                    const int startRow, const int stopRow){

    return 0;
}

void OnlineModel::trainOnline(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args &args, std::string output){
    std::cerr << "Training online for " << args.epochs << " ...\n";

    // Iterate over rows
    int rows = features.rows();
    int examples = rows * args.epochs;
    for(int i = 0; i < rows * args.epochs; ++i){
        r = i % rows;
        printProgress(i, examples);
        update(labels.row(r), labels.size(r), features.row(r), features.size(r), args);
    }

    save(args, output);
}
