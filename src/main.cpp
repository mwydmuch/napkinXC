/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 */

#include "args.h"
#include "types.h"
#include "base.h"
#include "pltree.h"
#include "utils.h"

void test(Args &args) {
    SRMatrix<Label> labels;
    SRMatrix<Feature> features;
    args.readData(labels, features);
    args.load(args.model + "/args.bin");

    PLTree tree;
    tree.load(args.model + "/tree.bin");

    std::cerr << "Loading base classifiers ...\n";
    std::vector<Base*> bases;
    for(int i = 0; i < tree.nodes(); ++i) {
        Base* base = new Base();
        base->load(args.model + "/node_" + std::to_string(i) + ".bin");
        bases.push_back(base);
        printProgress(i, tree.nodes());
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
