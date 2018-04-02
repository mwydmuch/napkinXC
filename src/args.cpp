/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 */

#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>
#include "args.h"
#include "linear.h"
#include "utils.h"

Args::Args() {
    // Input/output options
    input = "";
    model = "";
    header = true;
    hash = -1;

    // Training options
    threads = getCpuCount() - 1;
    eps = 0.1;
    solverType = L2R_LR_DUAL;
    bias = true;

    // Tree options
    tree = "";
    arity = 2;
    treeType = completeInOrder;

    // Prediction options
    topK = 1;

    // Private
    hFeatures = -1;
    hLabels = -1;
}

// Parse args
void Args::parseArgs(const std::vector<std::string>& args) {
    for (int ai = 2; ai < args.size(); ai += 2) {
        if (args[ai][0] != '-') {
            std::cerr << "Provided argument without a dash! Usage:" << std::endl;
            printHelp();
        }

        try {
            if (args[ai] == "-h" || args[ai] == "--help") {
                std::cerr << "Here is the help! Usage:" << std::endl;
                printHelp();
            }

            // Input/output options
            else if (args[ai] == "-i" || args[ai] == "--input")
                input = std::string(args.at(ai + 1));
            else if (args[ai] == "-m" || args[ai] == "--model")
                model = std::string(args.at(ai + 1));
            else if (args[ai] == "--header")
                header = std::stoi(args.at(ai + 1)) != 0;
            else if (args[ai] == "--hash")
                hash = std::stoi(args.at(ai + 1));

            // Training options
            else if (args[ai] == "-t" || args[ai] == "--threads"){
                threads = std::stoi(args.at(ai + 1));
                if(threads == 0) threads = getCpuCount();
                else if(threads == -1) threads = getCpuCount() - 1;
            } else if (args[ai] == "--eps")
                eps = std::stof(args.at(ai + 1));
            else if (args[ai] == "--bias")
                bias = std::stoi(args.at(ai + 1)) != 0;
            else if (args[ai] == "--solver") {
                if (args.at(ai + 1) == "L2R_LR") solverType = L2R_LR;
                else if (args.at(ai + 1) == "L2R_L2LOSS_SVC_DUAL") solverType = L2R_L2LOSS_SVC_DUAL;
                else if (args.at(ai + 1) == "L2R_L2LOSS_SVC") solverType = L2R_L2LOSS_SVC;
                else if (args.at(ai + 1) == "L2R_L1LOSS_SVC_DUAL") solverType = L2R_L1LOSS_SVC_DUAL;
                else if (args.at(ai + 1) == "L1R_L2LOSS_SVC") solverType = L1R_L2LOSS_SVC;
                else if (args.at(ai + 1) == "L1R_LR") solverType = L1R_LR;
                else if (args.at(ai + 1) == "L2R_LR_DUAL") solverType = L2R_LR_DUAL;
                else {
                    std::cerr << "Unknown solver type: " << args.at(ai + 1) << std::endl;
                    printHelp();
                }
            }

            // Tree options
            else if (args[ai] == "--tree")
                tree = std::string(args.at(ai + 1));
            else if (args[ai] == "--arity")
                arity = std::stoi(args.at(ai + 1));
            else if (args[ai] == "--treeType") {
                if (args.at(ai + 1) == "completeInOrder") treeType = completeInOrder;
                else if (args.at(ai + 1) == "completeRandom") treeType = completeRandom;
                else if (args.at(ai + 1) == "complete") treeType = complete;
                else {
                    std::cerr << "Unknown tree type: " << args.at(ai + 1) << std::endl;
                    printHelp();
                }
            }

            // Prediction options
            else if (args[ai] == "--topK")
                topK = std::stoi(args.at(ai + 1));
            else {
                std::cerr << "Unknown argument: " << args[ai] << std::endl;
                printHelp();
            }
        }
        catch (std::out_of_range) {
            std::cerr << args[ai] << " is missing an argument" << std::endl;
            printHelp();
        }
    }

    if (input.empty() || model.empty()) {
        std::cerr << "Empty input or model path." << std::endl;
        printHelp();
    }
}

// Reads train/test data to sparse matrix
// TODO: check if line is sorted, sort if not
void Args::readData(SRMatrix<Label>& labels, SRMatrix<Feature>& features){
    std::cerr << "Loading data from: " << input << std::endl;

    int hRows = -1;
    size_t nextPos, pos = 0;
    std::string line;

    std::ifstream in;
    in.open(input);

    // Read header (only for checks)
    // Format: #rows #features #labels
    if(header){
        getline(in, line);

        nextPos = line.find_first_of(" ", pos);
        hRows = std::stoi(line.substr(pos, nextPos - pos));
        pos = nextPos + 1;

        nextPos = line.find_first_of(" ", pos);
        if(hFeatures < 0) hFeatures = std::stoi(line.substr(pos, nextPos - pos));
        pos = nextPos + 1;

        nextPos = line.find_first_of(" ", pos);
        if(hLabels < 0) hLabels = std::stoi(line.substr(pos, nextPos - pos));

        std::cerr << "  Header: rows: " << hRows << ", features: " << hFeatures << ", labels: " << hLabels << std::endl;
    }

    // Read examples
    // LibSvm line format label,label,... feature(:value) feature(:value) ...
    while (getline(in, line)){
        std::vector<Label> lLabels;
        std::vector<Feature> lFeatures;
        pos = line[0] == ' ' ? 1 : 0;

        while((nextPos = line.find_first_of(",: ", pos))){
            // Label
            if ((pos == 0 || line[pos - 1] == ',') && (line[nextPos] == ',' || line[nextPos] == ' '))
                lLabels.push_back(std::stoi(line.substr(pos, nextPos - pos)));

            // Feature (LibLinear ignore feature 0)
            else if (line[pos - 1] == ' ' && line[nextPos] == ':')
                lFeatures.push_back({std::stoi(line.substr(pos, nextPos - pos)) + 1, 1.0});

            else if (line[pos - 1] == ':' && (line[nextPos] == ' ' || nextPos == std::string::npos))
                lFeatures.back().value = std::stof(line.substr(pos, nextPos - pos));

            if (nextPos == std::string::npos) break;
            pos = nextPos + 1;
        }

        // Norm row
        double norm = 0;
        for(int f = 0; f < lFeatures.size(); ++f)
            norm += lFeatures[f].value * lFeatures[f].value;
        norm = std::sqrt(norm);
        for(int f = 0; f < lFeatures.size(); ++f)
            lFeatures[f].value /= norm;

        // Add bias feature
        if(bias && hFeatures < 0)
            lFeatures.push_back({lFeatures.back().index + 1, 1.0});
        else if(bias)
            lFeatures.push_back({hFeatures + 1, 1.0});

        labels.appendRow(lLabels);
        features.appendRow(lFeatures);
    }

    in.close();

    if(bias && !header){
        for(int r = 0; r < features.rows(); ++r) {
            features.data()[r][features.sizes()[r] - 1].index = features.cols() - 1;
            features.data()[r][features.sizes()[r] - 1].value = 1.0;
        }
    }

    if(hLabels < 0) hLabels = labels.cols();
    if(hFeatures < 0) hFeatures = features.cols() - (bias ? 1 : 0);

    // Checks
    assert(labels.rows() == features.rows());
    if(header) assert(hRows == labels.rows());
    assert(hLabels >= labels.cols());
    assert(hFeatures + 1 + (bias ? 1 : 0) >= features.cols() );

//    for (int r = 0; r < features.rows(); ++r){
//        for(int c = 0; c < features.sizes()[r]; ++c)
//            std::cerr << features.data()[r][c].index << ":" << features.data()[r][c].value << " ";
//        std::cerr << "\n";
//    }

    std::cerr << "  Loaded: rows: " << labels.rows() << ", features: " << features.cols() - 1 - (bias ? 1 : 0) << ", labels: " << labels.cols() << std::endl;
}

// TODO
void Args::printHelp(){
    std::cerr << "HELP!\n";
    exit(EXIT_FAILURE);
}

void Args::save(std::string outfile){
    std::ofstream out;
    out.open(outfile);
    save(out);
    out.close();
}

void Args::save(std::ostream& out){
    out.write((char*) &bias, sizeof(bias));
    out.write((char*) &hFeatures, sizeof(hFeatures));
    out.write((char*) &hLabels, sizeof(hLabels));
    out.write((char*) &hash, sizeof(hash));
}

void Args::load(std::string infile){
    std::ifstream in;
    in.open(infile);
    load(in);
    in.close();
}

void Args::load(std::istream& in){
    in.read((char*) &bias, sizeof(bias));
    in.read((char*) &hFeatures, sizeof(hFeatures));
    in.read((char*) &hLabels, sizeof(hLabels));
    in.read((char*) &hash, sizeof(hash));
}
