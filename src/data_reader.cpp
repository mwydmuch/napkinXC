/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#include <algorithm>
#include <unordered_map>

#include "data_reader.h"
#include "libsvm_reader.h"
#include "vw_reader.h"
#include "misc.h"


std::shared_ptr<DataReader> DataReader::factory(Args& args) {
    std::shared_ptr<DataReader> dataReader = nullptr;
    switch (args.dataFormatType) {
    case libsvm: dataReader = std::static_pointer_cast<DataReader>(std::make_shared<LibSvmReader>()); break;
    case vw: dataReader = std::static_pointer_cast<DataReader>(std::make_shared<VowpalWabbitReader>()); break;
    default: throw std::invalid_argument("Unknown data reader type!");
    }

    return dataReader;
}

DataReader::DataReader() {
    hLabels = 0;
    hFeatures = 0;
    hRows = 0;
}

DataReader::~DataReader() {}

void DataReader::readHeader(std::string& line) {}

// Reads train/test data to sparse matrix
void DataReader::readData(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args) {
    std::cerr << "Loading data from: " << args.input << std::endl;

    std::ifstream in;
    in.open(args.input);
    std::string line;

    // Read header
    if (args.header && supportHeader) {
        getline(in, line);
        readHeader(line);
    }

    std::vector<Label> lLabels;
    std::vector<Feature> lFeatures;
    if (args.hash) hFeatures = args.hash;

    // Read examples
    if (!hRows) std::cerr << "  ?%\r";
    int i = 0;
    while (getline(in, line)) {
        if (hRows) printProgress(i++, hRows); // If the number of rows is know, print progress

        lLabels.clear();
        lFeatures.clear();

        readLine(line, lLabels, lFeatures);

        if (args.hash) {
            std::unordered_map<int, double> lHashed;
            for (auto& f : lFeatures) lHashed[hash(f.index) % args.hash] += f.value;

            lFeatures.clear();
            for (const auto& f : lHashed) lFeatures.push_back({f.first + 1, f.second});
        }

        // Check if it requires sorting
        if (!std::is_sorted(lFeatures.begin(), lFeatures.end())) sort(lFeatures.begin(), lFeatures.end());

        // Norm row
        if (args.norm) unitNorm(lFeatures);
        if (args.featuresThreshold > 0) threshold(lFeatures, args.featuresThreshold);

        // Add bias feature after applying norm
        if (args.bias && !hFeatures)
            lFeatures.push_back({lFeatures.back().index + 1, args.biasValue});
        else if (args.bias)
            lFeatures.push_back({hFeatures + 1, args.biasValue});

        // TODO: If bias will be feature with index = 1, then sorting won't be required

        labels.appendRow(lLabels);
        features.appendRow(lFeatures);
    }

    in.close();

    if (args.bias && !args.header) {
        for (int r = 0; r < features.rows(); ++r)
            features[r][features.size(r) - 1] = {features.cols() - 1, args.biasValue};
    }

    if (!hLabels) hLabels = labels.cols();
    if (!hFeatures) hFeatures = features.cols() - (args.bias ? 1 : 0);

    // Checks
    assert(labels.rows() == features.rows());
    // assert(hLabels >= labels.cols());
    // assert(hFeatures + 1 + (args.bias ? 1 : 0) >= features.cols());

    // Print data
    /*
    for (int r = 0; r < features.rows(); ++r){
       for(int c = 0; c < features.size(r); ++c)
           std::cerr << features.row(r)[c].index << ":" << features.row(r)[c].value << " ";
       std::cerr << "\n";
    }
    */

    // Print info about loaded data
    std::cerr << "  Loaded: rows: " << labels.rows() << ", features: " << features.cols() - 1 - (args.bias ? 1 : 0)
              << ", labels: " << labels.cols() << std::endl;
}

void DataReader::save(std::ostream& out) {
    out.write((char*)&hFeatures, sizeof(hFeatures));
    out.write((char*)&hLabels, sizeof(hLabels));
}

void DataReader::load(std::istream& in) {
    in.read((char*)&hFeatures, sizeof(hFeatures));
    in.read((char*)&hLabels, sizeof(hLabels));
}

void DataReader::printInfoAboutData(SRMatrix<Label>& labels, SRMatrix<Feature>& features){
    std::cout << "Data stats:"
              << "\n  Data points: " << features.rows()
              << "\n  Uniq features: " << features.cols()
              << "\n  Uniq labels: " << labels.cols()
              << "\n  Labels / data point: " << static_cast<double>(labels.cells()) / labels.rows()
              << "\n  Features / data point: " << static_cast<double>(features.cells()) / features.rows()
              << "\n  Data points / label: " << static_cast<double>(labels.cols()) / labels.cells()
              << "\n  Data points / feature: " << static_cast<double>(features.cols()) / features.cells()
              << "\n";
}