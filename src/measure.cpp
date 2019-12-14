/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#include <string>
#include <fstream>
#include <iomanip>
#include <mutex>
#include "model.h"
#include "threads.h"

#include "measure.h"
#include "set_utility.h"


std::vector<std::shared_ptr<Measure>> Measure::factory(Args& args, int outputSize){
    std::vector<std::shared_ptr<Measure>> measures;

    std::vector<std::string> measuresNames = split(args.measures, ',');
    for(const auto& m : measuresNames){
        // TODO: Add wrong values handling
        std::vector<std::string> mAt = split(m, '@');
        if(mAt.size() > 1){
            int k = std::stoi(mAt[1]);
            if(mAt[0] == "p")
                measures.push_back(std::static_pointer_cast<Measure>(std::make_shared<PrecisionAtK>(args, outputSize, k)));
            else if(mAt[0] == "r")
                measures.push_back(std::static_pointer_cast<Measure>(std::make_shared<RecallAtK>(args, outputSize, k)));
            else if(mAt[0] == "c")
                measures.push_back(std::static_pointer_cast<Measure>(std::make_shared<CoverageAtK>(args, outputSize, k)));
            else
                throw std::invalid_argument("Unknown measure type!");
        } else {
            if (m == "p")
                measures.push_back(std::static_pointer_cast<Measure>(std::make_shared<Precision>(args, outputSize)));
            else if (m == "r")
                measures.push_back(std::static_pointer_cast<Measure>(std::make_shared<Recall>(args, outputSize)));
            else if (m == "c")
                measures.push_back(std::static_pointer_cast<Measure>(std::make_shared<Coverage>(args, outputSize)));
            else if (m == "acc")
                measures.push_back(std::static_pointer_cast<Measure>(std::make_shared<Accuracy>(args, outputSize)));
            else if (m == "uP")
                measures.push_back(std::static_pointer_cast<Measure>(std::make_shared<PrecisionUtility>(args, outputSize)));
            else if (m == "uF1")
                measures.push_back(std::static_pointer_cast<Measure>(std::make_shared<F1Utility>(args, outputSize)));
            else if (m == "uAlfa")
                measures.push_back(std::static_pointer_cast<Measure>(std::make_shared<UtilityAlfa>(args, outputSize)));
            else if (m == "uAlfaBeta")
                measures.push_back(std::static_pointer_cast<Measure>(std::make_shared<UtilityAlfaBeta>(args, outputSize)));
            else if (m == "uDeltaGamma")
                measures.push_back(std::static_pointer_cast<Measure>(std::make_shared<UtilityDeltaGamma>(args, outputSize)));
            else if (m == "s")
                measures.push_back(std::static_pointer_cast<Measure>(std::make_shared<PredictionSize>(args, outputSize)));
            else
                throw std::invalid_argument("Unknown measure type!");
        }
    }

    return measures;
}

Measure::Measure(Args& args, int outputSize) {
    sum = 0;
    count = 0;
}

void Measure::accumulate(SRMatrix<Label>& labels, std::vector<std::vector<Prediction>>& predictions){
    assert(predictions.size() == labels.rows());
    for(int i = 0; i < labels.rows(); ++i) accumulate(labels[i], predictions[i]);
}

double Measure::value() {
    return sum / count;
}

MeasureAtK::MeasureAtK(Args& args, int outputSize, int k) : Measure(args, outputSize){
    this->k = k;
}

Recall::Recall(Args& args, int outputSize) : Measure(args, outputSize){
    name = "Recall";
}

void Recall::accumulate(Label* labels, const std::vector<Prediction>& prediction){
    double tp = 0;
    int l;
    for (const auto& p : prediction){
        l = -1;
        while(labels[++l] > -1)
            if (p.label == labels[l]){
                ++tp;
                break;
            }
    }
    l = -1;
    while(labels[++l] > -1);
    if(l > 0) {
        sum += tp / l;
        ++count;
    }
}

RecallAtK::RecallAtK(Args& args, int outputSize, int k) : MeasureAtK(args, outputSize, k){
    name = "R@" + std::to_string(k);
}

void RecallAtK::accumulate(Label* labels, const std::vector<Prediction>& prediction){
    double tp = 0;
    int l;
    for (int i = 0; i < k; ++i){
        l = -1;
        while(labels[++l] > -1)
            if (prediction[i].label == labels[l]){
                ++tp;
                break;
            }
    }
    l = -1;
    while(labels[++l] > -1);
    if(l > 0) {
        sum += tp / l;
        ++count;
    }
}

Precision::Precision(Args& args, int outputSize) : Measure(args, outputSize){
    name = "Precision";
}

void Precision::accumulate(Label* labels, const std::vector<Prediction>& prediction){
    double tp = 0;
    int l;
    for (const auto& p : prediction){
        l = -1;
        while(labels[++l] > -1)
            if (p.label == labels[l]){
                ++tp;
                break;
            }
    }
    sum += tp / prediction.size();
    ++count;
}

PrecisionAtK::PrecisionAtK(Args& args, int outputSize, int k) : MeasureAtK(args, outputSize, k){
    name = "P@" + std::to_string(k);
}

void PrecisionAtK::accumulate(Label* labels, const std::vector<Prediction>& prediction){
    double tp = 0;
    int l;
    for (int i = 0; i < k; ++i){
        l = -1;
        while(labels[++l] > -1)
            if (prediction[i].label == labels[l]){
                ++tp;
                break;
            }
    }
    sum += tp / k;
    ++count;
}

Coverage::Coverage(Args& args, int outputSize) : Measure(args, outputSize){
    name = "Coverage";
    m = outputSize;
}

void Coverage::accumulate(Label* labels, const std::vector<Prediction>& prediction){
    int l;
    for (const auto& p : prediction){
        l = -1;
        while(labels[++l] > -1)
            if (p.label == labels[l]){
                seen.insert(p.label);
                break;
            }
    }
}

double Coverage::value(){
    return static_cast<double>(seen.size()) / m;
}

CoverageAtK::CoverageAtK(Args& args, int outputSize, int k) : MeasureAtK(args, outputSize, k){
    name = "C@" + std::to_string(k);
    m = outputSize;
}

void CoverageAtK::accumulate(Label* labels, const std::vector<Prediction>& prediction){
    int l;
    for (int i = 0; i < k; ++i){
        l = -1;
        while(labels[++l] > -1)
            if (prediction[i].label == labels[l]){
                seen.insert(prediction[i].label);
                break;
            }
    }
}

double CoverageAtK::value(){
    return static_cast<double>(seen.size()) / m;
}

Accuracy::Accuracy(Args& args, int outputSize) : Measure(args, outputSize){
    name = "Acc";
}

void Accuracy::accumulate(Label* labels, const std::vector<Prediction>& prediction){
    if(labels[0] > -1 && prediction.size() && labels[0] == prediction[0].label)
        ++sum;
    ++count;
}

PredictionSize::PredictionSize(Args& args, int outputSize) : Measure(args, outputSize){
    name = "Mean prediction size";
}

void PredictionSize::accumulate(Label* labels, const std::vector<Prediction>& prediction){
    sum += prediction.size();
    ++count;
}
