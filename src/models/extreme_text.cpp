/*
 Copyright (c) 2020 by Marek Wydmuch

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

#include "extreme_text.h"
#include "threads.h"


ExtremeText::ExtremeText() {
    type = extremeText;
    name = "extremeText";
}

void ExtremeText::trainThread(int threadId, ExtremeText* model, SRMatrix& labels,
                                    SRMatrix& features, Args& args, const int startRow, const int stopRow) {
    const int rowsRange = stopRow - startRow;
    const int examples = rowsRange * args.epochs;
    Real loss = 0;
    for (int i = 0; i < examples; ++i) {
        Real lr = args.eta * (1.0 - (static_cast<Real>(i) / examples));
        if (!threadId) printProgress(i, examples, lr, loss / i);

        int r = startRow + i % rowsRange;
        loss += model->update(lr, features[r], labels[r], args);
    }
}

Real ExtremeText::updateNode(TreeNode* node, Real label, Vector& hidden, Vector& gradient, Real lr, Real l2){
    size_t i = node->index;

    //Real pred = 1.0 / (1.0 + std::exp(-dotVectors(outputW[i], hidden)));
    Real val = outputW[i].dot(hidden);
    Real pred = sigmoid(val);
    Real grad = label - pred;
    //Log(COUT) << val << " " << pred << " " << grad << "\n";

    for(int j = 0; j < dims; ++j){
        gradient[j] += lr * (grad * outputW[i][j] - l2 * gradient[j]);
        outputW[i][j] += lr * (grad * hidden[j] - l2 * outputW[i][j]);
    }

    return label ? -log(pred) : -log(1.0 - pred);
}

Real ExtremeText::update(Real lr, const SparseVector& features, const SparseVector& labels, const Args& args){

    // Compute hidden
    Real valuesSum = 0;
    Vector hidden(dims);
    for(auto &f : features){
        valuesSum += f.value;
        hidden.add(inputW[f.index], f.value);
    }
    hidden.div(valuesSum);

    // Gather nodes to update
    UnorderedSet<TreeNode*> nPositive;
    UnorderedSet<TreeNode*> nNegative;

    getNodesToUpdate(nPositive, nNegative, labels);

    // Compute gradient
    Vector gradient(dims);
    //Real lr = 0.5 * args.eta * std::sqrt(1.0 / ++t);
    Real loss = 0.0;
    for (auto &n : nPositive)
        loss += updateNode(n, 1.0, hidden, gradient, lr, args.l2Penalty);

    for (auto &n : nNegative)
        loss += updateNode(n, 0.0, hidden, gradient, lr, args.l2Penalty);

    // Update input weights
    gradient.div(valuesSum);
    for(auto &f : features)
        inputW[f.index].add(gradient, f.value);

    return loss;
}

void ExtremeText::train(SRMatrix& labels, SRMatrix& features, Args& args, std::string output) {

    // Create tree
    if (!tree) {
        tree = new LabelTree();
        tree->buildTreeStructure(labels, features, args);
    }
    m = tree->getNumberOfLeaves();

    dims = args.dims;
    inputW = RMatrix<Vector>(features.cols(), dims);

    std::default_random_engine rng(args.getSeed());
    std::uniform_real_distribution<Real> dist(-1.0 / dims, 1.0 / dims);

    for(int i = 0; i < inputW.rows(); ++i)
        for(int j = 0; j < inputW.cols(); ++j) inputW[i][j] = dist(rng);

    outputW = RMatrix<Vector>(tree->size(), dims);

    // Iterate over rows
    Log(CERR) << "Training extremeText for " << args.epochs << " epochs in " << args.threads << " threads ...\n";

    ThreadSet tSet;
    int tRows = ceil(static_cast<Real>(features.rows()) / args.threads);
    for (int t = 0; t < args.threads; ++t)
        tSet.add(trainThread, t, this, std::ref(labels), std::ref(features), std::ref(args), t * tRows,
                 std::min((t + 1) * tRows, features.rows()));
    tSet.joinAll();

    // Save training output
    tree->saveToFile(joinPath(output, "tree.bin"));
    tree->saveTreeStructure(joinPath(output, "tree"));

    std::ofstream out(joinPath(output, "XTWeights.bin"));
    inputW.save(out);
    outputW.save(out);
    out.close();
}

void ExtremeText::load(Args& args, std::string infile) {
    Log(CERR) << "Loading " << name << " model ...\n";

    tree = new LabelTree();
    tree->loadFromFile(joinPath(infile, "tree.bin"));

    std::ifstream in(joinPath(infile, "XTWeights.bin"));
    inputW.load(in);
    outputW.load(in);
    in.close();

    assert(inputW.cols() == outputW.cols());
    dims = inputW.cols();

    assert(tree->size() == outputW.rows());
    m = tree->getNumberOfLeaves();

    loaded = true;
}

void ExtremeText::predict(std::vector<Prediction>& prediction, SparseVector& features, Args& args){
    auto hidden = computeHidden(features);
    PLT::predict(prediction, hidden, args);
}

Real ExtremeText::predictForLabel(Label label, SparseVector& features, Args& args){
    auto hidden = computeHidden(features);
    Real value = PLT::predictForLabel(label, hidden, args);
    return value;
}

SparseVector ExtremeText::computeHidden(const SparseVector& features){
    SparseVector hidden(dims, dims);
    for(size_t i = 0; i < dims; ++i) hidden.insertD(i, 0);

    Real valuesSum = 0;
    for(auto &f : features){
        valuesSum += f.value;
        hidden.add(inputW[f.index], f.value);
    }

    hidden.div(valuesSum);

    return hidden;
}