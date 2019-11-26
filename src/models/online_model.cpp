/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#include "online_model.h"
#include "threads.h"


void onlineTrainThread(int threadId, OnlineModel* model, SRMatrix<Label>& labels, SRMatrix<Feature>& features,
                       Args& args, const int startRow, const int stopRow){
    const int rowsRange = stopRow - startRow;
    const int examples = rowsRange * args.epochs;
    for(int i = 0; i < examples; ++i){
        if(!threadId) printProgress(i, examples);
        int r = startRow + i % rowsRange;
        model->update(labels.row(r), labels.size(r), features.row(r), features.size(r), args);
    }
}

void OnlineModel::train(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args &args, std::string output) {
    std::cerr << "Preparing online model ...\n";

    // Init model
    init(labels.cols(), args);

    // Iterate over rows
    std::cerr << "Training online for " << args.epochs << " epochs in " << args.threads << " threads ...\n";

    ThreadSet tSet;
    int tRows = ceil(static_cast<double>(features.rows())/args.threads);
    for(int t = 0; t < args.threads; ++t)
        tSet.add(onlineTrainThread, t, this, std::ref(labels), std::ref(features), std::ref(args),
                 t * tRows, std::min((t + 1) * tRows, features.rows()));
    tSet.joinAll();

    // Save traning output
    save(args, output);
}
