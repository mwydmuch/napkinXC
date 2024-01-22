/*
 Copyright (c) 2019-2021 by Marek Wydmuch, Kalina Jasinska-Kobus

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


#pragma once

#include <fstream>
#include <future>
#include <string>

#include "args.h"
#include "base.h"
#include "basic_types.h"
#include "misc.h"

class Model {
public:
    static std::shared_ptr<Model> factory(Args& args);

    Model();
    virtual ~Model();

    virtual void train(SRMatrix& labels, SRMatrix& features, Args& args, std::string output) = 0;
    virtual void predict(std::vector<Prediction>& prediction, SparseVector& features, Args& args) = 0;
    virtual Real predictForLabel(Label label, SparseVector& features, Args& args) = 0;
    virtual std::vector<std::vector<Prediction>> predictBatch(SRMatrix& features, Args& args);

    // Prediction with thresholds and ofo
    virtual void setThresholds(std::vector<Real> th);
    virtual void updateThresholds(UnorderedMap<int, Real> thToUpdate);
    std::vector<Real> getThresholds(){ return thresholds; };

    virtual void setLabelsWeights(std::vector<Real> lw);
    std::vector<Real> getLabelsWeights(){ return labelsWeights; };

    std::vector<Real> ofo(SRMatrix& features, SRMatrix& labels, Args& args);
    Real microOfo(SRMatrix& features, SRMatrix& labels, Args& args);
    std::vector<Real> macroOfo(SRMatrix& features, SRMatrix& labels, Args& args);

    virtual void load(Args& args, std::string infile) = 0;
    virtual void preload(Args& args, std::string infile) { preloaded = true; };
    virtual void unload() { preloaded = false; loaded = false; };
    bool isPreloaded() { return preloaded; };
    bool isLoaded() { return loaded; };

    virtual void printInfo() {}
    inline int outputSize() { return m; };

protected:
    ModelType type;
    std::string name;
    int m; // Output size/number of labels
    int f; // Input size/number of features
    bool preloaded;
    bool loaded;
    std::vector<Real> thresholds; // For prediction with thresholds
    std::vector<Real> labelsWeights; // For prediction with label weights

    // Base utils
    static Base* trainBase(ProblemData& problemsData, Args& args);
    static void trainBatchThread(std::vector<std::promise<Base *>>& results, std::vector<ProblemData>& problemsData, Args& args, int threadId, int threads);
    static void trainBases(const std::string& outfile, std::vector<ProblemData>& problemsData, Args& args);
    static void trainBases(std::ofstream& out, std::vector<ProblemData>& problemsData, Args& args);

    static void saveResults(std::ofstream& out, std::vector<std::future<Base*>>& results, bool saveGrads=false);
    static std::vector<Base*> loadBases(const std::string& infile, bool resume=false, RepresentationType loadAs=map);

private:
    static void predictBatchThread(int threadId, Model* model, std::vector<std::vector<Prediction>>& predictions,
                                   SRMatrix& features, Args& args, const int startRow, const int stopRow);

    static void macroOfoThread(int threadId, Model* model, std::vector<Real>& as, std::vector<Real>& bs,
                               SRMatrix& features, SRMatrix& labels, Args& args,
                               const int startRow, const int stopRow);
};
