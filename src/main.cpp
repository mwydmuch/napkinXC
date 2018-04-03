/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 */

#include "args.h"
#include "types.h"
#include "base.h"
#include "pltree.h"
#include "utils.h"
#include "threads.h"

Base* nodeLoadThread(std::string nodeFile){
    Base* base = new Base();
    base->load(nodeFile);
    return base;
}

void test(Args &args) {
    SRMatrix<Label> labels;
    SRMatrix<Feature> features;
    args.readData(labels, features);
    args.load(args.model + "/args.bin");

    PLTree tree;
    tree.load(args.model + "/tree.bin");

    std::cerr << "Loading base classifiers ...\n";
    std::vector<Base*> bases;

    if(args.threads > 1){
        // Run loading in parallel
        ThreadPool tPool(args.threads);
        std::vector<std::future<Base*>> results;

        for (int i = 0; i < tree.nodes(); ++i)
            results.emplace_back(tPool.enqueue(nodeLoadThread, args.model + "/node_" + std::to_string(i) + ".bin"));

        // Get loaded classfiers
        for(int i = 0; i < results.size(); ++i) {
            Base* base = results[i].get();
            bases.push_back(base);
            printProgress(i, results.size());
        }
    } else {
        for(int i = 0; i < tree.nodes(); ++i) {
            Base* base = new Base();
            base->load(args.model + "/node_" + std::to_string(i) + ".bin");
            bases.push_back(base);
            printProgress(i, tree.nodes());
        }
    }

    tree.test(labels, features, bases, args);

    for(auto base : bases)
        delete base;
}

void train(Args &args) {
    SRMatrix<Label> labels;
    SRMatrix<Feature> features;
    args.readData(labels, features);

    PLTree tree;
    tree.train(labels, features, args);
}

int main(int argc, char** argv) {
    std::vector<std::string> arg(argv, argv + argc);
    Args args = Args();
    args.parseArgs(arg);

    if (arg.size() < 2)
        args.printHelp();

    std::string command(arg[1]);
    if(command == "train")
        train(args);
    else if(command == "test")
        test(args);
    else
        args.printHelp();

    return 0;
}
