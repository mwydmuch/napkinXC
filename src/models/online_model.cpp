/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#include "online_model.h"


//int onlineTrainThread(int threadId, Model* model, SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args,
//                    const int startRow, const int stopRow){
//
//    return 0;
//}

void OnlineModel::train(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args &args, std::string output){
    std::cerr << "Preparing online model ...\n";

    // Init model
    init(labels.cols(), args);

    // Iterate over rows
    std::cerr << "Training online for " << args.epochs << " epochs ...\n";

    // One thread version
    int rows = features.rows();
    int examples = rows * args.epochs;
    for(int i = 0; i < rows * args.epochs; ++i){
        int r = i % rows;
        printProgress(i, examples);
        update(labels.row(r), labels.size(r), features.row(r), features.size(r), args);
    }

    // Save traning output
    save(args, output);
}
