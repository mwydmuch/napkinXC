/**
 * Copyright (c) 2020 by Marek Wydmuch
 * All rights reserved.
 */

#include "extreme_text.h"
#include "threads.h"


ExtremeText::ExtremeText() {
    type = extremeText;
    name = "extremeText";
}

void ExtremeText::trainThread(int threadId, ExtremeText* model, SRMatrix<Label>& labels,
                                    SRMatrix<Feature>& features, Args& args, const int startRow, const int stopRow) {
    const int rowsRange = stopRow - startRow;
    const int examples = rowsRange * args.epochs;
    double loss = 0;
    for (int i = 0; i < examples; ++i) {
        double lr = args.eta * (1.0 - (static_cast<double>(i) / examples));
        if (!threadId) printProgress(i, examples, lr, loss / i);

        int r = startRow + i % rowsRange;
        loss += model->update(lr, features[r], labels[r], labels.size(r), args);
    }
}

double ExtremeText::updateNode(TreeNode* node, double label, Vector<XTWeight>& hidden, Vector<XTWeight>& gradient, double lr, double l2){
    size_t i = node->index;

    //double pred = 1.0 / (1.0 + std::exp(-dotVectors(outputW[i], hidden)));
    double val = dotVectors(outputW[i], hidden);
    double pred = sigmoid(dotVectors(outputW[i], hidden));
    double grad = label - pred;
    std::cout << val << " " << pred << " " << grad << "\n";
    if(isnan(pred))
        exit(1);

    for(int j = 0; j < dims; ++j){
        gradient[j] += lr * (grad * outputW[i][j] - l2 * gradient[j]);
        outputW[i][j] += lr * (grad * hidden[j] - l2 * outputW[i][j]);
    }

    return label ? -log(pred) : -log(1.0 - pred);
}

double ExtremeText::update(double lr, Feature* features, Label* labels, int rSize, Args& args){

    // Compute hidden
    double valuesSum = 0;
    Vector<XTWeight> hidden(dims, 0);
    for(Feature* f = features; f->index != -1; ++f){
        valuesSum += f->value;
        addVector(inputW[f->index], f->value, hidden);
    }
    divVector(hidden, valuesSum);

    // Gather nodes to update
    std::unordered_set<TreeNode*> nPositive;
    std::unordered_set<TreeNode*> nNegative;

    getNodesToUpdate(nPositive, nNegative, labels, rSize);

    // Compute gradient
    Vector<XTWeight> gradient(dims, 0);
    //double lr = 0.5 * args.eta * std::sqrt(1.0 / ++t);
    double loss = 0.0;
    for (auto &n : nPositive)
        loss += updateNode(n, 1.0, hidden, gradient, lr, args.l2Penalty);

    for (auto &n : nNegative)
        loss += updateNode(n, 0.0, hidden, gradient, lr, args.l2Penalty);

    // Update input weights
    divVector(gradient, valuesSum);
    for(Feature* f = features; f->index != -1; ++f)
        addVector(gradient, f->value, inputW[f->index]);

    return loss;
}

void ExtremeText::train(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args, std::string output) {

    // Create tree
    if (!tree) {
        tree = new Tree();
        tree->buildTreeStructure(labels, features, args);
    }
    m = tree->getNumberOfLeaves();

    dims = args.dims;
    inputW = Matrix<XTWeight>(features.cols(), dims);

    std::default_random_engine rng(args.getSeed());
    std::uniform_real_distribution<double> dist(-1.0 / dims, 1.0 / dims);

    for(int i = 0; i < inputW.rows(); ++i)
        for(int j = 0; j < inputW.cols(); ++j) inputW[i][j] = dist(rng);

    outputW = Matrix<XTWeight>(tree->t, dims);

    // Iterate over rows
    std::cerr << "Training extremeText for " << args.epochs << " epochs in " << args.threads << " threads ...\n";

    ThreadSet tSet;
    int tRows = ceil(static_cast<double>(features.rows()) / args.threads);
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
    std::cerr << "Loading " << name << " model ...\n";

    tree = new Tree();
    tree->loadFromFile(joinPath(infile, "tree.bin"));

    std::ifstream in(joinPath(infile, "XTWeights.bin"));
    inputW.load(in);
    outputW.load(in);
    in.close();

    assert(inputW.cols() == outputW.cols());
    dims = inputW.cols();

    assert(tree->t == outputW.rows());
    m = tree->getNumberOfLeaves();

    if(!args.thresholds.empty())
        tree->populateNodeLabels();
}

void ExtremeText::predict(std::vector<Prediction>& prediction, Feature* features, Args& args){
    Feature* hidden = computeHidden(features);
    PLT::predict(prediction, hidden, args);
    delete[] hidden;
}

void ExtremeText::predictWithThresholds(std::vector<Prediction>& prediction, Feature* features, std::vector<float>& thresholds,
                           Args& args){
    Feature* hidden = computeHidden(features);
    PLT::predictWithThresholds(prediction, hidden, thresholds, args);
    delete[] hidden;
}

double ExtremeText::predictForLabel(Label label, Feature* features, Args& args){
    Feature* hidden = computeHidden(features);
    double value = PLT::predictForLabel(label, hidden, args);
    delete[] hidden;
    return value;
}

Feature* ExtremeText::computeHidden(Feature* features){
    Feature* hidden = new Feature[dims + 1];
    for(size_t i = 0; i < dims; ++i) {
        hidden[i].index = i;
        hidden[i].value = 0;
    }
    hidden[dims].index = -1;

    double valuesSum = 0;
    for(Feature* f = features; f->index != -1; ++f){
        valuesSum += f->value;
        for(size_t i = 0; i < dims; ++i) hidden[i].value += inputW[f->index][i] * f->value;
    }

    for(size_t i = 0; i < dims; ++i)
        hidden[i].value /= valuesSum;

    return hidden;
}