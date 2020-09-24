/*
 Copyright (c) 2019-2020 by Marek Wydmuch

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 */

#include "online_model.h"
#include "threads.h"
#include "resources.h"
#include "log.h"


void OnlineModel::onlineTrainThread(int threadId, OnlineModel* model, SRMatrix<Label>& labels,
                                    SRMatrix<Feature>& features, Args& args, const int startRow, const int stopRow) {
    const int rowsRange = stopRow - startRow;
    const int examples = rowsRange * args.epochs;
    for (int i = 0; i < examples; ++i) {
        if (!threadId) printProgress(i, examples);
        int r = startRow + i % rowsRange;
        model->update(r, labels.row(r), labels.size(r), features.row(r), features.size(r), args);

        if(!threadId && logLevel >= CERR_DEBUG && i % (examples / 100) == 0){
            auto res = getResources();
            Log(COUT) << "  R mem (MB): " << res.currentRealMem / 1024
                      << ", V mem (MB): " << res.currentVirtualMem / 1024
                      << ", R mem peak (MB): " << res.peakRealMem / 1024
                      << ", V mem peak (MB): " << res.peakVirtualMem / 1024 << "\n";
        }
    }
}

void OnlineModel::train(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args, std::string output) {
    Log(CERR) << "Preparing online model ...\n";

    // Init model
    if(args.resume) load(args, output);
    else init(labels, features, args);

    // Iterate over rows
    Log(CERR) << "Training online for " << args.epochs << " epochs in " << args.threads << " threads ...\n";

    ThreadSet tSet;
    int tRows = ceil(static_cast<double>(features.rows()) / args.threads);
    for (int t = 0; t < args.threads; ++t)
        tSet.add(onlineTrainThread, t, this, std::ref(labels), std::ref(features), std::ref(args), t * tRows,
                 std::min((t + 1) * tRows, features.rows()));
    tSet.joinAll();

    // Save training output
    save(args, output);
}
