/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 */

#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>
#include <iomanip>

#include "args.h"
#include "linear.h"
#include "utils.h"

Args::Args() {
    command = "";
    seed = time(0);

    // Input/output options
    input = "";
    model = "";
    header = true;
    hash = 0;
    bias = true;
    biasValue = 1.0;
    norm = true;
    maxFeatures = -1;
    projectDim = 100;

    // Training options
    threads = getCpuCount();
    eps = 0.1;
    cost = 10.0;
    solverType = L2R_LR_DUAL;
    solverName = "L2R_LR_DUAL";
    labelsWeights = true;
    optimizerName = "libliner";
    optimizerType = libliner;
    iter = 50;
    eta = 0.5;
    threshold = 0.1;

    // Tree options
    treeStructure = "";
    arity = 2;
    //treeType = completeInOrder;
    //treeTypeName = "completeInOrder";
    treeType = hierarchicalKMeans;
    treeTypeName = "hierarchicalKMeans";
    maxLeaves = 100;

    // K-Means tree options
    kMeansEps = 0.001;
    kMeansBalanced = true;

    // Prediction options
    topK = 1;
    sparseWeights = true;

    // KNN options
    kNN = 0;
    kNNMaxFreq = 25;

    // Private
    hFeatures = 0;
    hLabels = 0;
}

// Parse args
void Args::parseArgs(const std::vector<std::string>& args) {
    command = args[1];

    if(command != "train" && command != "test" && command != "shrink"){
        std::cerr << "Unknown command type: " << command << std::endl;
        printHelp();
    }

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
            else if (args[ai] == "--seed")
                seed = std::stoi(args.at(ai + 1));

            // Input/output options
            else if (args[ai] == "-i" || args[ai] == "--input")
                input = std::string(args.at(ai + 1));
            else if (args[ai] == "-m" || args[ai] == "--model")
                model = std::string(args.at(ai + 1));
            else if (args[ai] == "--header")
                header = std::stoi(args.at(ai + 1)) != 0;
            else if (args[ai] == "--bias")
                bias = std::stoi(args.at(ai + 1)) != 0;
            else if (args[ai] == "--norm")
                norm = std::stoi(args.at(ai + 1)) != 0;
            else if (args[ai] == "--hash")
                hash = std::stoi(args.at(ai + 1));
            else if (args[ai] == "--threshold")
                threshold = std::stof(args.at(ai + 1));
            else if (args[ai] == "--eta")
                eta = std::stof(args.at(ai + 1));
            else if (args[ai] == "--iter")
                iter = std::stoi(args.at(ai + 1));
            else if (args[ai] == "--maxFeatures")
                maxFeatures = std::stoi(args.at(ai + 1));
            else if (args[ai] == "--projectDim")
                projectDim = std::stoi(args.at(ai + 1));

            // Training options
            else if (args[ai] == "-t" || args[ai] == "--threads"){
                threads = std::stoi(args.at(ai + 1));
                if(threads == 0) threads = getCpuCount();
                else if(threads == -1) threads = getCpuCount() - 1;
            } else if (args[ai] == "-e" || args[ai] == "--eps")
                eps = std::stof(args.at(ai + 1));
            else if (args[ai] == "-c" || args[ai] == "-C" || args[ai] == "--cost")
                cost = std::stof(args.at(ai + 1));
            else if (args[ai] == "--labelsWeights")
                labelsWeights = std::stoi(args.at(ai + 1)) != 0;
            else if (args[ai] == "--solver") {
                solverName = args.at(ai + 1);
                if (args.at(ai + 1) == "L2R_LR_DUAL") solverType = L2R_LR_DUAL;
                else if (args.at(ai + 1) == "L2R_LR") solverType = L2R_LR;
                else if (args.at(ai + 1) == "L1R_LR") solverType = L1R_LR;
                else if (args.at(ai + 1) == "L2R_L2LOSS_SVC_DUAL") solverType = L2R_L2LOSS_SVC_DUAL;
                else if (args.at(ai + 1) == "L2R_L2LOSS_SVC") solverType = L2R_L2LOSS_SVC;
                else if (args.at(ai + 1) == "L2R_L1LOSS_SVC_DUAL") solverType = L2R_L1LOSS_SVC_DUAL;
                else if (args.at(ai + 1) == "L1R_L2LOSS_SVC") solverType = L1R_L2LOSS_SVC;
                else {
                    std::cerr << "Unknown solver type: " << args.at(ai + 1) << std::endl;
                    printHelp();
                }
            }
            else if (args[ai] == "--optimizer") {
                optimizerName = args.at(ai + 1);
                if (args.at(ai + 1) == "liblinear") optimizerType = libliner;
                else if (args.at(ai + 1) == "sgd") optimizerType = sgd;
                else{
                    std::cerr << "Unknown optimizer type: " << args.at(ai + 1) << std::endl;
                    printHelp();
                }
            }
            else if (args[ai] == "-e" || args[ai] == "--eta")
                eta = std::stof(args.at(ai + 1));
            else if (args[ai] == "--iter")
                iter = std::stoi(args.at(ai + 1));

            // Tree options
            else if (args[ai] == "-a" || args[ai] == "--arity")
                arity = std::stoi(args.at(ai + 1));
            else if (args[ai] == "--maxLeaves")
                maxLeaves = std::stoi(args.at(ai + 1));
            else if (args[ai] == "--kMeansEps")
                kMeansEps = std::stof(args.at(ai + 1));
            else if (args[ai] == "--kMeansBalanced")
                kMeansBalanced = std::stoi(args.at(ai + 1)) != 0;
            else if (args[ai] == "--treeStructure")
                treeStructure = std::string(args.at(ai + 1));
            else if (args[ai] == "--treeType") {
                treeTypeName = args.at(ai + 1);
                if (args.at(ai + 1) == "completeInOrder") treeType = completeInOrder;
                else if (args.at(ai + 1) == "completeRandom") treeType = completeRandom;
                else if (args.at(ai + 1) == "balancedInOrder") treeType = balancedInOrder;
                else if (args.at(ai + 1) == "balancedRandom") treeType = balancedRandom;
                else if (args.at(ai + 1) == "hierarchicalKMeans") treeType = hierarchicalKMeans;
                else if (args.at(ai + 1) == "kMeansWithProjection" ) treeType = kMeansWithProjection;
                else if (args.at(ai + 1) == "topDown") treeType = topDown;
                else if (args.at(ai + 1) == "huffman") treeType = huffman;
                else if (args.at(ai + 1) == "leaveFreqBehind") treeType = leaveFreqBehind;
                else if (args.at(ai + 1) == "kMeansHuffman") treeType = kMeansHuffman;
                else {
                    std::cerr << "Unknown tree type: " << args.at(ai + 1) << std::endl;
                    printHelp();
                }
            }

            // Prediction options
            else if (args[ai] == "--topK")
                topK = std::stoi(args.at(ai + 1));
            else if (args[ai] == "--sparseWeights")
                sparseWeights = std::stoi(args.at(ai + 1)) != 0;
            else if (args[ai] == "--kNN")
                kNN = std::stoi(args.at(ai + 1));
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
void Args::readData(SRMatrix<Label>& labels, SRMatrix<Feature>& features){
    std::cerr << "Loading data from: " << input << std::endl;

    std::ifstream in;
    in.open(input);

    int hRows = -1;
    std::string line;

    // Read header
    // Format: #rows #features #labels
    // TODO: add validation
    if(header){
        size_t nextPos, pos = 0;
        getline(in, line);

        nextPos = line.find_first_of(" ", pos);
        hRows = std::stoi(line.substr(pos, nextPos - pos));
        pos = nextPos + 1;

        nextPos = line.find_first_of(" ", pos);
        if(!hFeatures) hFeatures = std::stoi(line.substr(pos, nextPos - pos));
        pos = nextPos + 1;

        nextPos = line.find_first_of(" ", pos);
        if(!hLabels) hLabels = std::stoi(line.substr(pos, nextPos - pos));

        std::cerr << "  Header: rows: " << hRows << ", features: " << hFeatures << ", labels: " << hLabels << std::endl;
    }

    if(hash != 0) hFeatures = hash;

    std::vector<Label> lLabels;
    std::vector<Feature> lFeatures;

    // Read examples
    while (getline(in, line)){
        lLabels.clear();
        lFeatures.clear();

        readLine(line, lLabels, lFeatures);

        labels.appendRow(lLabels);
        features.appendRow(lFeatures);
    }

    in.close();

    if(bias && !header){
        for(int r = 0; r < features.rows(); ++r) {
            features.data()[r][features.sizes()[r] - 1].index = features.cols() - 1;
            features.data()[r][features.sizes()[r] - 1].value = biasValue;
        }
    }

    if(!hLabels) hLabels = labels.cols();
    if(!hFeatures) hFeatures = features.cols() - (bias ? 1 : 0);

    // Checks
    assert(labels.rows() == features.rows());
    if(header) assert(hRows == labels.rows());
    assert(hLabels >= labels.cols());
    assert(hFeatures + 1 + (bias ? 1 : 0) >= features.cols() );

    // Print data
    /*
    for (int r = 0; r < features.rows(); ++r){
       for(int c = 0; c < features.size(r); ++c)
           std::cerr << features.row(r)[c].index << ":" << features.row(r)[c].value << " ";
       std::cerr << "\n";
    }
    */

    std::cerr << "  Loaded: rows: " << labels.rows() << ", features: " << features.cols() - 1 - (bias ? 1 : 0) << ", labels: " << labels.cols() << std::endl;
}

// Reads line in LibSvm format label,label,... feature(:value) feature(:value) ...
void Args::readLine(std::string& line, std::vector<Label>& lLabels, std::vector<Feature>& lFeatures){
    size_t nextPos, pos = line[0] == ' ' ? 1 : 0;
    bool requiresSort = false;

    while((nextPos = line.find_first_of(",: ", pos))){
        // Label
        if((pos == 0 || line[pos - 1] == ',') && (line[nextPos] == ',' || line[nextPos] == ' '))
            lLabels.push_back(std::stoi(line.substr(pos, nextPos - pos)));

        // Feature (LibLinear ignore feature 0)
        else if(line[pos - 1] == ' ' && line[nextPos] == ':')
            lFeatures.push_back({std::stoi(line.substr(pos, nextPos - pos)) + 1, 1.0});

        else if(line[pos - 1] == ':' && (line[nextPos] == ' ' || nextPos == std::string::npos))
            lFeatures.back().value = std::stof(line.substr(pos, nextPos - pos));

        if(nextPos == std::string::npos) break;
        pos = nextPos + 1;
    }

    // Select subset of most important features
    /*
    if(maxFeatures > 0) {
        std::sort(lFeatures.rbegin(), lFeatures.rend());
        lFeatures.resize(std::min(100, static_cast<int>(lFeatures.size())));
    }
     */

    // Norm row
    if(norm) unitNorm(lFeatures);

    // Add bias feature
    if(bias && hFeatures < 0)
        lFeatures.push_back({lFeatures.back().index + 1, biasValue});
    else if(bias)
        lFeatures.push_back({hFeatures + 1, biasValue});
}

void Args::printArgs(){
    if (command == "train" || command == "test"){
        std::cerr << "napkinXML - " << command
            << "\n  Input: " << input
            << "\n    Header: " << header << ", bias: " << bias << ", norm: " << norm << ", hash: " << hash
            << "\n  Model: " << model
            << "\n    Optimizer: " << optimizerName;
        if(optimizerType == libliner)
            std::cerr << "\n    LibLinear: Solver: " << solverName << ", eps: " << eps << ", cost: " << cost << ", threshold: " << threshold;
        else if(optimizerType == sgd)
            std::cerr << "\n    SGD: eta: " << eta << ", iter: " << iter << ", threshold: " << threshold;
        if(treeStructure.empty()) {
            std::cerr << "\n    Tree type: " << treeTypeName << ", arity: " << arity;
            if (treeType == hierarchicalKMeans) std::cerr << ", k-means eps: " << kMeansEps << ", balanced: " << kMeansBalanced;
            if (treeType == hierarchicalKMeans || treeType == balancedInOrder || treeType == balancedRandom)
                std::cerr << ", max leaves: " << maxLeaves;
        }
        std::cerr << "\n  Threads: " << threads << "\n";
    }
    else if (command == "shrink")
        std::cerr << "napkinXML - " << command
            << "\n  Input model: " << input
            << "\n  Output model: " << model
            << "\n  Threshold: " << threshold << "\n";
}

void Args::printHelp(){
    std::cerr << R"HELP(Usage: nxml <command> <args>

    Commands:
        train
        test

    Args:
        General:
        -i, --input         Input dataset in LibSvm format
        -m, --model         Model's dir
        -t, --threads       Number of threads used for training and testing (default = -1)
                            Note: -1 to use #cpus - 1, 0 to use #cpus
        --header            Input contains header (default = 1)
                            Header format: #lines #features #labels
        --hash              Size of hashing space (default = 0)
                            Note: 0 to disable
        --seed              Model's seed

        Base classifiers:
        --optimizer         Use LibLiner or SGD (default = libliner)
                            Optimizers: liblinear, sgd
        --bias              Add bias term (default = 1)
        --labelsWeights     Increase the weight of minority labels in base classifiers (default = 1)

        LibLinear:
        -s, --solver        LibLinear solver (default = L2R_LR_DUAL)
                            Supported solvers: L2R_LR_DUAL, L2R_LR, L1R_LR,
                                               L2R_L2LOSS_SVC_DUAL, L2R_L2LOSS_SVC, L2R_L1LOSS_SVC_DUAL, L1R_L2LOSS_SVC
                            See: https://github.com/cjlin1/liblinear
        -c, -C, --cost      Inverse of regularization strength. Must be a positive float.
                            Like in support vector machines, smaller values specify stronger
                            regularization. (default = 10.0)
                            Note: -1 to automatically find best value for each node.
        -e, --eps           Stopping criteria (default = 0.1)
                            See: https://github.com/cjlin1/liblinear

        SGD:
        -e, --eta           Step size of SGD
        --iter              Number of epochs of SGD

        Tree:
        -a, --arity         Arity of a tree (default = 2)
        --maxLeaves         Maximum number of leaves (labels) in one internal node. (default = 100)
        --tree              File with tree structure
        --treeType          Type of a tree to build if file with structure is not provided
                            Tree types: hierarchicalKMeans, huffman, completeInOrder, completeRandom,
                                        balancedInOrder, balancedRandom,

        K-Means tree:
        --kMeansEps         Stopping criteria for K-Means clustering (default = 0.001)
        --kMeansBalanced    Use balanced K-Means clustering (default = 1)

        Random projection:
        --projectDim        Number or random direction

        K-NNs:
        --kNN               Number of nearest neighbors used for prediction
    )HELP";
    exit(EXIT_FAILURE);
}

void Args::save(std::string outfile){
    std::ofstream out(outfile);
    save(out);
    out.close();
}

void Args::save(std::ostream& out){
    //TODO: save names and other parameters that are displayed
    out.write((char*) &hFeatures, sizeof(hFeatures));
    out.write((char*) &hLabels, sizeof(hLabels));
    out.write((char*) &bias, sizeof(bias));
    out.write((char*) &norm, sizeof(norm));
    out.write((char*) &hash, sizeof(hash));
}

void Args::load(std::string infile){
    std::ifstream in(infile);
    load(in);
    in.close();
}

void Args::load(std::istream& in){
    in.read((char*) &hFeatures, sizeof(hFeatures));
    in.read((char*) &hLabels, sizeof(hLabels));
    in.read((char*) &bias, sizeof(bias));
    in.read((char*) &norm, sizeof(norm));
    in.read((char*) &hash, sizeof(hash));
}
