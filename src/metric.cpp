/*
 Copyright (c) 2019-2020 by Marek Wydmuch

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 */

#include "model.h"
#include "threads.h"
#include <fstream>
#include <iomanip>
#include <mutex>
#include <string>

#include "metric.h"


std::vector<std::shared_ptr<Metric>> Metric::factory(Args& args, int outputSize) {
    std::vector<std::shared_ptr<Metric>> metrics;

    std::vector<std::string> metricsNames = split(toLower(args.metrics), ',');
    for (const auto& m : metricsNames) {
        // TODO: Add wrong values handling
        std::vector<std::string> mAt = split(m, '@');
        if (mAt.size() > 1) {
            int k = std::stoi(mAt[1]);
            if (mAt[0] == "p" || mAt[0] == "precision")
                metrics.push_back(std::static_pointer_cast<Metric>(std::make_shared<PrecisionAtK>(k)));
            else if (mAt[0] == "r" || mAt[0] == "recall")
                metrics.push_back(std::static_pointer_cast<Metric>(std::make_shared<RecallAtK>(k)));
            else if (mAt[0] == "dcg")
                metrics.push_back(std::static_pointer_cast<Metric>(std::make_shared<DCGAtK>(k)));
            else if (mAt[0] == "ndcg")
                metrics.push_back(std::static_pointer_cast<Metric>(std::make_shared<NDCGAtK>(k)));
            else if (mAt[0] == "c" || mAt[0] == "coverage")
                metrics.push_back(std::static_pointer_cast<Metric>(std::make_shared<CoverageAtK>(outputSize, k)));
            else if (mAt[0] == "tp")
                metrics.push_back(std::static_pointer_cast<Metric>(std::make_shared<TruePositivesAtK>(k)));
            else
                throw std::invalid_argument("Unknown measure type: " + mAt[0]);
        } else {
            if (m == "p" || m =="precision")
                metrics.push_back(std::static_pointer_cast<Metric>(std::make_shared<Precision>()));
            else if (m == "r" || m =="recall")
                metrics.push_back(std::static_pointer_cast<Metric>(std::make_shared<Recall>()));
            else if (m == "samplef1")
                metrics.push_back(std::static_pointer_cast<Metric>(std::make_shared<SampleF1>()));
            else if (m == "microf1")
                metrics.push_back(std::static_pointer_cast<Metric>(std::make_shared<MicroF1>()));
            else if (m == "macrof1")
                metrics.push_back(std::static_pointer_cast<Metric>(std::make_shared<MacroF1>(outputSize)));
            else if (m == "c" || m == "coverage")
                metrics.push_back(std::static_pointer_cast<Metric>(std::make_shared<Coverage>(outputSize)));
            else if (m == "acc" || m == "accuracy")
                metrics.push_back(std::static_pointer_cast<Metric>(std::make_shared<Accuracy>()));
            else if (m == "s" || m == "size")
                metrics.push_back(std::static_pointer_cast<Metric>(std::make_shared<PredictionSize>()));
            else if (m == "hl")
                metrics.push_back(std::static_pointer_cast<Metric>(std::make_shared<HammingLoss>()));
            else if (m == "tp")
                metrics.push_back(std::static_pointer_cast<Metric>(std::make_shared<TruePositives>()));
            else if (m == "fp")
                metrics.push_back(std::static_pointer_cast<Metric>(std::make_shared<FalsePositives>()));
            else if (m == "fn")
                metrics.push_back(std::static_pointer_cast<Metric>(std::make_shared<FalseNegatives>()));
            else
                throw std::invalid_argument("Unknown measure type: " + m + "!");
        }
    }

    return metrics;
}

Metric::Metric() {
    sum = 0;
    sumSq = 0;
    count = 0;
}

void Metric::accumulate(SRMatrix& labels, std::vector<std::vector<Prediction>>& predictions) {
    assert(predictions.size() == labels.rows());
    for (int i = 0; i < labels.rows(); ++i) accumulate(labels[i], predictions[i]);
}

double Metric::value() { return sum / count; }

double Metric::stdDev() {
    double m = mean();
    return sumSq / count - m * m;
}

void Metric::addValue(double value){
    sum += value;
    sumSq += value * value;
    ++count;
}

MetricAtK::MetricAtK(int k) : k(k) {
    if (k < 1) throw std::invalid_argument("K cannot be lower then 1!");
}

TruePositivesAtK::TruePositivesAtK(int k) : MetricAtK(k) {
    name = "TP@";
    meanMetric = true;
}

void TruePositivesAtK::accumulate(SparseVector& labels, const std::vector<Prediction>& prediction) {
    addValue(calculate(labels, prediction, k));
}

double TruePositivesAtK::calculate(SparseVector& labels, const std::vector<Prediction>& prediction, int k) {
    double tp = 0;
    for (int i = 0; i < std::min(k, static_cast<int>(prediction.size())); ++i) {
        for(auto &l : labels)
            if (prediction[i].label == l.index) {
                ++tp;
                break;
            }
    }

    return tp;
}

TruePositives::TruePositives() {
    name = "TP";
    meanMetric = true;
}

void TruePositives::accumulate(SparseVector& labels, const std::vector<Prediction>& prediction) {
    addValue(TruePositives::calculate(labels, prediction));
}

double TruePositives::calculate(SparseVector& labels, const std::vector<Prediction>& prediction) {
    return TruePositivesAtK::calculate(labels, prediction, prediction.size());
}

FalsePositives::FalsePositives() {
    name = "FP";
    meanMetric = true;
}

void FalsePositives::accumulate(SparseVector& labels, const std::vector<Prediction>& prediction) {
    addValue(FalsePositives::calculate(labels, prediction));
}

double FalsePositives::calculate(SparseVector& labels, const std::vector<Prediction>& prediction) {
    double fp = 0;

    for (const auto& p : prediction) {
        bool found = false;
        for(auto &l : labels) {
            if (p.label == l.index) {
                found = true;
                break;
            }
        }
        if (!found) ++fp;
    }

    return fp;
}

FalseNegatives::FalseNegatives() {
    name = "FN";
    meanMetric = true;
}

void FalseNegatives::accumulate(SparseVector& labels, const std::vector<Prediction>& prediction) {
    addValue(FalseNegatives::calculate(labels, prediction));
}

double FalseNegatives::calculate(SparseVector& labels, const std::vector<Prediction>& prediction) {
    double fn = 0;

    
    for(auto &l : labels) {
        bool found = false;
        for (const auto& p : prediction) {
            if (p.label == l.index) {
                found = true;
                break;
            }
        }
        if (!found) ++fn;
    }
    return fn;
}


Recall::Recall() {
    name = "Recall";
    meanMetric = true;
}

void Recall::accumulate(SparseVector& labels, const std::vector<Prediction>& prediction) {
    double tp = TruePositives::calculate(labels, prediction);
    if(labels.nonZero()) addValue(tp / labels.nonZero());
}

RecallAtK::RecallAtK(int k) : MetricAtK(k) {
    name = "R@" + std::to_string(k);
    meanMetric = true;
}

void RecallAtK::accumulate(SparseVector& labels, const std::vector<Prediction>& prediction) {
    double tp = TruePositivesAtK::calculate(labels, prediction, k);
    if(labels.nonZero()) addValue(tp / labels.nonZero());
}

Precision::Precision() {
    name = "Precision";
    meanMetric = true;
}

void Precision::accumulate(SparseVector& labels, const std::vector<Prediction>& prediction) {
    double tp = TruePositives::calculate(labels, prediction);
    if (!prediction.empty()) addValue(tp / prediction.size());
}

PrecisionAtK::PrecisionAtK(int k) : MetricAtK(k) {
    name = "P@" + std::to_string(k);
    meanMetric = true;
}

void PrecisionAtK::accumulate(SparseVector& labels, const std::vector<Prediction>& prediction) {
    addValue(TruePositivesAtK::calculate(labels, prediction, k) / k);
}

DCGAtK::DCGAtK(int k) : MetricAtK(k) {
    name = "DCG@" + std::to_string(k);
    meanMetric = true;
}

void DCGAtK::accumulate(SparseVector& labels, const std::vector<Prediction>& prediction) {
    addValue(calculate(labels, prediction, k));
}

double DCGAtK::calculate(SparseVector& labels, const std::vector<Prediction>& prediction, int k){
    double score = 0;
    for (int i = 0; i < std::min(k, static_cast<int>(prediction.size())); ++i) {
        for(auto &l : labels)
            if (prediction[i].label == l.index) {
                score += 1.0 / std::log2(i + 2);
                break;
            }
    }

    return score;
}


NDCGAtK::NDCGAtK(int k) : MetricAtK(k) {
    name = "nDCG@" + std::to_string(k);
    meanMetric = true;
}

void NDCGAtK::accumulate(SparseVector& labels, const std::vector<Prediction>& prediction) {
    double nDenominator = 0;

    int i = 0;
    for(auto &l : labels){
        nDenominator += 1.0 / std::log2(i + 2);
        ++i;
        if(i >= k) break;
    }

    if (labels.nonZero() > 0) addValue(DCGAtK::calculate(labels, prediction, k) / nDenominator);
    else addValue(0);

}

Coverage::Coverage(int outputSize) : m(outputSize) {
    name = "Coverage";
    meanMetric = false;
}

void Coverage::accumulate(SparseVector& labels, const std::vector<Prediction>& prediction) {
    for (const auto& p : prediction) {
        for(auto &l : labels)
            if (p.label == l.index) {
                seen.insert(p.label);
                break;
            }
    }
}

double Coverage::value() { return static_cast<double>(seen.size()) / m; }

CoverageAtK::CoverageAtK(int outputSize, int k) : MetricAtK(k), m(outputSize) {
    name = "C@" + std::to_string(k);
    meanMetric = false;
}

void CoverageAtK::accumulate(SparseVector& labels, const std::vector<Prediction>& prediction) {
    for (int i = 0; i < std::min(k, static_cast<int>(prediction.size())); ++i) {
        for(auto &l : labels)
            if (prediction[i].label == l.index) {
                seen.insert(prediction[i].label);
                break;
            }
    }
}

double CoverageAtK::value() { return static_cast<double>(seen.size()) / m; }

Accuracy::Accuracy() {
    name = "Acc";
    meanMetric = true;
}

void Accuracy::accumulate(SparseVector& labels, const std::vector<Prediction>& prediction) {
    if (!prediction.empty() && labels[0] == prediction[0].label) addValue(1);
    else addValue(0);
}

PredictionSize::PredictionSize() {
    name = "Prediction size";
    meanMetric = true;
}

void PredictionSize::accumulate(SparseVector& labels, const std::vector<Prediction>& prediction) {
    addValue(prediction.size());
}

HammingLoss::HammingLoss() {
    name = "Hamming loss";
    meanMetric = true;
}

void HammingLoss::accumulate(SparseVector& labels, const std::vector<Prediction>& prediction) {
    addValue(FalsePositives::calculate(labels, prediction) + FalseNegatives::calculate(labels, prediction));
}

SampleF1::SampleF1() {
    name = "Sample-F1";
    meanMetric = true;
}

void SampleF1::accumulate(SparseVector& labels, const std::vector<Prediction>& prediction) {
    double tp = TruePositives::calculate(labels, prediction);
    if (!prediction.empty() && labels.nonZero() > 0) {
        double p = tp / prediction.size();
        double r = tp / labels.nonZero();
        if(p > 0 && r > 0) addValue(2 * p * r / (p + r));
    }
}

MicroF1::MicroF1() {
    name = "Micro-F1";
    meanMetric = false;
}

void MicroF1::accumulate(SparseVector& labels, const std::vector<Prediction>& prediction) {
    double tp = TruePositives::calculate(labels, prediction);
    sum += 2 * tp;
    count += 2 * tp + FalsePositives::calculate(labels, prediction) + FalseNegatives::calculate(labels, prediction);
}

MacroF1::MacroF1(int outputSize) : m(outputSize), zeroDivisionDenominator(1) {
    name = "Macro-F1";
    meanMetric = false;
    labelsTP.resize(m, 0);
    labelsFP.resize(m, 0);
    labelsFN.resize(m, 0);
}

void MacroF1::accumulate(SparseVector& labels, const std::vector<Prediction>& prediction){

    for (const auto& p : prediction) {
        bool found = false;
        for(auto &l : labels) {
            if (p.label == l.index) {
                found = true;
                ++labelsTP[p.label];
                break;
            }
        }
        if (!found) ++labelsFP[p.label];
    }

    for(auto &l : labels) {
        bool found = false;
        for (const auto& p : prediction) {
            if (p.label == l.index) {
                found = true;
                break;
            }
        }
        if (!found) ++labelsFN[l.index];
    }
}

double MacroF1::value(){
    double sum = 0;
    for(int i = 0; i < m; ++i){
        double denominator = 2 * labelsTP[i] + labelsFP[i] + labelsFN[i];
        denominator = (denominator > 0) ? 2 * labelsTP[i] / denominator : zeroDivisionDenominator;
        sum += denominator;
    }
    return sum / m;
}
