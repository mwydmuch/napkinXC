/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 */

#include "args.h"
#include "types.h"
#include "base.h"
#include "pltree.h"
#include "utils.h"

void train(Args &args) {
    SRMatrix<Label> labels;
    SRMatrix<Feature> features;

    // Load train data
    args.printArgs();
    args.readData(labels, features);

    // Train and save tree
    PLTree tree;
    tree.train(labels, features, args);

    // Save args
    args.save(joinPath(args.model, "args.bin"));
}

void test(Args &args) {
    SRMatrix<Label> labels;
    SRMatrix<Feature> features;

    // Load args and test data
    args.load(joinPath(args.model, "args.bin"));
    args.printArgs();
    args.readData(labels, features);

    // Load tree
    PLTree tree;
    tree.load(joinPath(args.model, "tree.bin"));
    tree.test(labels, features, args);
}

void predict(Args &args) {
    // Load args
    args.load(joinPath(args.model, "args.bin"));
    args.printArgs();

    PLTree tree;
    tree.load(joinPath(args.model, "tree.bin"));

    // Predict data from cin and output to cout
    if(args.input == "-"){
        //TODO
    }

    // Read data from file and output prediction to output
    else {
        SRMatrix<Label> labels;
        SRMatrix<Feature> features;
        args.readData(labels, features);
        //TODO
    }
}

void buildTree(Args &args) {
    args.printArgs();

    SRMatrix<Label> labels;
    SRMatrix<Feature> features;
    args.readData(labels, features);

    PLTree tree;
    tree.buildTreeStructure(labels, features, args);
}

void shrink(Args &args) {
    args.printArgs();

    PLTree tree;
    tree.load(args.input + "/tree.bin");

    std::ifstream in(args.input + "/weights.bin");
    std::ofstream out(args.model + "/weights.bin");
    for (int i = 0; i < tree.nodes(); ++i){
        Base base;
        base.load(in, args);
        base.threshold(args.threshold);
        base.save(out, args);
    }
    in.close();
    out.close();
}

int main(int argc, char** argv) {
    std::vector<std::string> arg(argv, argv + argc);
    Args args = Args();

    // Parse args
    args.parseArgs(arg);

    if(args.command == "train")
        train(args);
    else if(args.command == "test")
        test(args);
    else if(args.command == "predict")
        predict(args);
    else if(args.command == "tree")
        buildTree(args);
    else if(args.command == "shrink")
        shrink(args);
    else
        args.printHelp();

    return 0;
}
