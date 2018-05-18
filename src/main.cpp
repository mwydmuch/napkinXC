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
    args.load(joinPath(args.model, "args.bin"));
    args.printArgs();
    args.readData(labels, features);

    PLTree tree;
    tree.load(joinPath(args.model, "tree.bin"));
    tree.test(labels, features, args);
}

void train(Args &args) {
    args.printArgs();

    SRMatrix<Label> labels;
    SRMatrix<Feature> features;
    args.readData(labels, features);

    PLTree tree;
    tree.train(labels, features, args);
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
    args.parseArgs(arg);

    if(args.command == "train")
        train(args);
    else if(args.command == "test")
        test(args);
    else if(args.command == "tree")
        buildTree(args);
    else if(args.command == "shrink")
        shrink(args);
    else
        args.printHelp();

    return 0;
}
