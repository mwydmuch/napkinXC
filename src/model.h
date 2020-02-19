/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include <fstream>
#include <future>
#include <string>

#include "args.h"
#include "base.h"
#include "types.h"

class Model {
public:
    static std::shared_ptr<Model> factory(Args& args);

    Model();
    virtual ~Model();

    virtual void train(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args, std::string output) = 0;
    virtual void predict(std::vector<Prediction>& prediction, Feature* features, Args& args) = 0;
    virtual void predictWithThresholds(std::vector<Prediction>& prediction, Feature* features,
                                       std::vector<float>& thresholds, Args& args);
    virtual std::vector<float> ofo(SRMatrix<Feature>& features, SRMatrix<Label>& labels, Args& args);
    virtual double predictForLabel(Label label, Feature* features, Args& args) = 0;
    virtual std::vector<std::vector<Prediction>> predictBatch(SRMatrix<Feature>& features, Args& args);
    virtual std::vector<std::vector<Prediction>> predictBatchWithThresholds(SRMatrix<Feature>& features,
                                                                            std::vector<float>& thresholds, Args& args);

    virtual void load(Args& args, std::string infile) = 0;

    virtual void printInfo() {}
    inline int outputSize() { return m; };

protected:
    ModelType type;
    std::string name;
    int m; // Output size/number of labels

    // Base utils
    static Base* trainBase(int n, std::vector<double>& baseLabels, std::vector<Feature*>& baseFeatures,
                           std::vector<double>* instancesWeights, Args& args);

    static void saveResults(std::ofstream& out, std::vector<std::future<Base*>>& results);

    static void trainBases(std::string outfile, int n, std::vector<std::vector<double>>& baseLabels,
                           std::vector<std::vector<Feature*>>& baseFeatures,
                           std::vector<std::vector<double>*>* instancesWeights, Args& args);

    static void trainBases(std::ofstream& out, int n, std::vector<std::vector<double>>& baseLabels,
                           std::vector<std::vector<Feature*>>& baseFeatures,
                           std::vector<std::vector<double>*>* instancesWeights, Args& args);

    static void trainBasesWithSameFeatures(std::string outfile, int n, std::vector<std::vector<double>>& baseLabels,
                                           std::vector<Feature*>& baseFeatures,
                                           std::vector<double>* instancesWeights, Args& args);

    static void trainBasesWithSameFeatures(std::ofstream& out, int n, std::vector<std::vector<double>>& baseLabels,
                                           std::vector<Feature*>& baseFeatures,
                                           std::vector<double>* instancesWeights, Args& args);

    static std::vector<Base*> loadBases(std::string infile);

private:
    static void predictBatchThread(int threadId, Model* model, std::vector<std::vector<Prediction>>& predictions,
                                   SRMatrix<Feature>& features, Args& args, const int startRow, const int stopRow);

    static void predictBatchWithThresholdsThread(int threadId, Model* model,
                                                 std::vector<std::vector<Prediction>>& predictions,
                                                 SRMatrix<Feature>& features, std::vector<float>& thresholds,
                                                 Args& args, const int startRow, const int stopRow);

    static void ofoThread(int threadId, Model* model, std::vector<float>& thresholds,
                          std::vector<float>& as, std::vector<float>& bs,
                          SRMatrix<Feature>& features, SRMatrix<Label>& labels, Args& args,
                          const int startRow, const int stopRow);
};
