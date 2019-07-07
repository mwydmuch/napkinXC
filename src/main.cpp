/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 */

#include "args.h"
#include "data_reader.h"
#include "model.h"
#include "types.h"
#include "utils.h"


void train(Args &args) {
    SRMatrix<Label> labels;
    SRMatrix<Feature> features;

    args.printArgs();
    makeDir(args.output);
    args.saveToFile(joinPath(args.output, "args.bin"));

    // Create data reader and load train data
    std::shared_ptr<DataReader> reader = dataReaderFactory(args);
    reader->readData(labels, features, args);
    reader->saveToFile(joinPath(args.output, "data_reader.bin"));

    // Create and train model (train function also saves model)
    std::shared_ptr<Model> model = modelFactory(args);
    model->train(labels, features, args, args.output);
    model->printInfo();

    std::cerr << "All done!\n";
}

void test(Args &args) {
    SRMatrix<Label> labels;
    SRMatrix<Feature> features;

    // Load model args
    args.loadFromFile(joinPath(args.output, "args.bin"));
    args.printArgs();

    // Create data reader and load test data
    std::shared_ptr<DataReader> reader = dataReaderFactory(args);
    reader->loadFromFile(joinPath(args.output, "data_reader.bin"));
    reader->readData(labels, features, args);

    // Load model and test
    std::shared_ptr<Model> model = modelFactory(args);
    model->load(args, args.output);
    model->test(labels, features, args);
    model->printInfo();

    std::cerr << "All done!\n";
}

/*
void predict(Args &args) {
    // Load args
    args.load(joinPath(args.output, "args.bin"));
    args.printArgs();

    PLTree tree;
    tree.load(joinPath(args.output, "tree.bin"));

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
    std::ofstream out(args.output + "/weights.bin");
    for (int i = 0; i < tree.nodes(); ++i){
        Base base;
        base.load(in, args);
        base.threshold(args.threshold);
        base.save(out, args);
    }
    in.close();
    out.close();
}
 */

int main(int argc, char** argv) {
    std::vector<std::string> arg(argv, argv + argc);
    Args args = Args();

    // Parse args
    args.parseArgs(arg);

    if(args.command == "train")
        train(args);
    else if(args.command == "test")
        test(args);
    else
        args.printHelp();

    return 0;
}
