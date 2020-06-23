/**
 * Copyright (c) 2019-2020 by Marek Wydmuch, Kalina Jasinska-Kobus
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
    virtual double predictForLabel(Label label, Feature* features, Args& args) = 0;
    virtual std::vector<std::vector<Prediction>> predictBatch(SRMatrix<Feature>& features, Args& args);

    // Prediction with thresholds and ofo
    virtual void setThresholds(std::vector<double> th);
    virtual void updateThresholds(UnorderedMap<int, double> thToUpdate);
    virtual void predictWithThresholds(std::vector<Prediction>& prediction, Feature* features, Args& args);
    virtual std::vector<std::vector<Prediction>> predictBatchWithThresholds(SRMatrix<Feature>& features, Args& args);
    std::vector<double> ofo(SRMatrix<Feature>& features, SRMatrix<Label>& labels, Args& args);
    double microOfo(SRMatrix<Feature>& features, SRMatrix<Label>& labels, Args& args);
    std::vector<double> macroOfo(SRMatrix<Feature>& features, SRMatrix<Label>& labels, Args& args);

    virtual void load(Args& args, std::string infile) = 0;

    virtual void printInfo() {}
    inline int outputSize() { return m; };

protected:
    ModelType type;
    std::string name;
    int m; // Output size/number of labels
    std::vector<double> thresholds; // For prediction with thresholds

    // Base utils
    static Base* trainBase(int n, int r, std::vector<double>& baseLabels, std::vector<Feature*>& baseFeatures,
                           std::vector<double>* instancesWeights, Args& args);

    static void trainBatchThread(int n, int r, std::vector<std::promise<Base *>>& results,
                                 std::vector<std::vector<double>>& baseLabels,
                                 std::vector<std::vector<Feature*>>& baseFeatures,
                                 std::vector<std::vector<double>*>* instancesWeights,
                                 Args& args, int threadId, int threads);

    static void trainBases(std::string outfile, int n, std::vector<std::vector<double>>& baseLabels,
                           std::vector<std::vector<Feature*>>& baseFeatures,
                           std::vector<std::vector<double>*>* instancesWeights, Args& args);

    static void trainBases(std::ofstream& out, int n, std::vector<std::vector<double>>& baseLabels,
                           std::vector<std::vector<Feature*>>& baseFeatures,
                           std::vector<std::vector<double>*>* instancesWeights, Args& args);

    static void trainBatchWithSameFeaturesThread(int n, std::vector<std::promise<Base *>>& results,
                                                 std::vector<std::vector<double>>& baseLabels,
                                                 std::vector<Feature*>& baseFeatures,
                                                 std::vector<double>* instancesWeights,
                                                 Args& args, int threadId, int threads);

    static void trainBasesWithSameFeatures(std::string outfile, int n, std::vector<std::vector<double>>& baseLabels,
                                           std::vector<Feature*>& baseFeatures,
                                           std::vector<double>* instancesWeights, Args& args);

    static void trainBasesWithSameFeatures(std::ofstream& out, int n, std::vector<std::vector<double>>& baseLabels,
                                           std::vector<Feature*>& baseFeatures,
                                           std::vector<double>* instancesWeights, Args& args);

    static void saveResults(std::ofstream& out, std::vector<std::future<Base*>>& results);

    static std::vector<Base*> loadBases(std::string infile);

private:
    static void predictBatchThread(int threadId, Model* model, std::vector<std::vector<Prediction>>& predictions,
                                   SRMatrix<Feature>& features, Args& args, const int startRow, const int stopRow);

    static void predictBatchWithThresholdsThread(int threadId, Model* model, std::vector<std::vector<Prediction>>& predictions,
                                                 SRMatrix<Feature>& features, Args& args, const int startRow, const int stopRow);

    static void ofoThread(int threadId, Model* model, std::vector<double>& as, std::vector<double>& bs,
                          SRMatrix<Feature>& features, SRMatrix<Label>& labels, Args& args,
                          const int startRow, const int stopRow);
};
