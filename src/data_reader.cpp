/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#include <algorithm>
#include <unordered_map>

#include "data_reader.h"
#include "utils.h"
#include "libsvm_reader.h"


std::shared_ptr<DataReader> dataReaderFactory(Args &args){
    std::shared_ptr<DataReader> dataReader = nullptr;
    switch (args.dataFormatType) {
        case libsvm :
            dataReader = std::static_pointer_cast<DataReader>(std::make_shared<LibSvmReader>());
            break;
    }

    return dataReader;
}

DataReader::DataReader(){
    hLabels = 0;
    hFeatures = 0;
    hRows = 0;
}

DataReader::~DataReader(){}

// Reads train/test data to sparse matrix
void DataReader::readData(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args){
    std::cerr << "Loading data from: " << args.input << std::endl;

    std::ifstream in;
    in.open(args.input);
    std::string line;

    // Read header
    if(args.header){
        getline(in, line);
        readHeader(line);
    }

    std::vector<Label> lLabels;
    std::vector<Feature> lFeatures;

    // Read examples
    if(!hRows) std::cerr << "  ?%\r";
    int i = 0;
    while (getline(in, line)){
        if(hRows) printProgress(i, hRows); // If the number of rows is know, print progress

        lLabels.clear();
        lFeatures.clear();

        readLine(line, lLabels, lFeatures);

        if(args.hash) {
            std::unordered_map<int, double> lHashed;
            for(auto &f : lFeatures)
                lHashed[hash(f.index) % args.hash] += f.value;

            lFeatures.clear();
            for(const auto &f : lHashed)
                lFeatures.push_back({f.first + 1, f.second});
        }

        // Check if it requires sorting
        if(!std::is_sorted(lFeatures.begin(), lFeatures.end()))
            sort(lFeatures.begin(), lFeatures.end());

        // Norm row
        if(args.norm) unitNorm(lFeatures);
        if(args.featuresThreshold > 0) threshold(lFeatures, args.featuresThreshold);

        // Add bias feature after applying norm
        if(args.bias && !hFeatures) lFeatures.push_back({lFeatures.back().index + 1, args.biasValue});
        else if(args.bias) lFeatures.push_back({hFeatures + 1, args.biasValue});

        labels.appendRow(lLabels);
        features.appendRow(lFeatures);
    }

    in.close();

    if(args.bias && !args.header){
        for(int r = 0; r < features.rows(); ++r) {
            features.data()[r][features.sizes()[r] - 1].index = features.cols() - 1;
            features.data()[r][features.sizes()[r] - 1].value = args.biasValue;
        }
    }

    if(!hLabels) hLabels = labels.cols();
    if(!hFeatures) hFeatures = features.cols() - (args.bias ? 1 : 0);

    // Checks
    assert(labels.rows() == features.rows());
    //assert(hLabels >= labels.cols());
    //assert(hFeatures + 1 + (args.bias ? 1 : 0) >= features.cols());

    // Print data
    /*
    for (int r = 0; r < features.rows(); ++r){
       for(int c = 0; c < features.size(r); ++c)
           std::cerr << features.row(r)[c].index << ":" << features.row(r)[c].value << " ";
       std::cerr << "\n";
    }
    */

    // Print info about loaded data
    std::cerr << "  Loaded: rows: " << labels.rows() << ", features: " << features.cols() - 1 - (args.bias ? 1 : 0) << ", labels: " << labels.cols() << std::endl;

    // Print data stats
    std::cerr << "Data stats:"
              << "\n  # data points: " << features.rows()
              << "\n  # uniq features: " << features.cols()
              << "\n  # uniq labels: " << labels.cols()
              << "\n  Mean # labels per data point: " << static_cast<double>(labels.cells()) / labels.rows()
              << "\n  Mean # features per data point: " << static_cast<double>(features.cells()) / features.rows()
              << "\n  Mean # data points per label: " << static_cast<double>(labels.cols()) / labels.cells()
              << "\n  Mean # data points per feature: " << static_cast<double>(features.cols()) / features.cells()
              << "\n";
}

void DataReader::save(std::ostream& out){
    out.write((char*) &hFeatures, sizeof(hFeatures));
    out.write((char*) &hLabels, sizeof(hLabels));
}

void DataReader::load(std::istream& in){
    in.read((char*) &hFeatures, sizeof(hFeatures));
    in.read((char*) &hLabels, sizeof(hLabels));
}
