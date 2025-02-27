/*
 Copyright (c) 2018-2021 by Marek Wydmuch, Kalina Jasinska-Kobus, Robert Istvan Busa-Fekete

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

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <list>
#include <utility>
#include <vector>

#include "plt.h"


PLT::PLT() {
    nodeEvaluationCount = 0;
    nodeUpdateCount = 0;
    dataPointCount = 0;
    type = plt;
    name = "PLT";
    tree = nullptr;
}

void PLT::unload() {
    for (auto b : bases) delete b;
    bases.clear();
    bases.shrink_to_fit();
    tree = nullptr;
    Model::unload();
}

void PLT::assignDataPoints(std::vector<std::vector<Real>>& binLabels, std::vector<std::vector<Feature*>>& binFeatures,
                           std::vector<std::vector<Real>>& binWeights, SRMatrix& labels, SRMatrix& features, Args& args) {
    Log(CERR) << "Assigning data points to nodes ...\n";

    // Positive and negative nodes
    UnorderedSet<TreeNode*> nPositive;
    UnorderedSet<TreeNode*> nNegative;

    // Gather examples for each node
    int rows = features.rows();
    for (int r = 0; r < rows; ++r) {
        printProgress(r, rows);

        nPositive.clear();
        nNegative.clear();

        getNodesToUpdate(nPositive, nNegative, labels[r]);
        addNodesLabelsAndFeatures(binLabels, binFeatures, nPositive, nNegative, features[r]);

        nodeUpdateCount += nPositive.size() + nNegative.size();
        ++dataPointCount;
    }

    unsigned long long usedMem = nodeUpdateCount * (sizeof(Real) + sizeof(Feature*)) + binLabels.size() * (sizeof(binLabels) + sizeof(binFeatures));
    Log(CERR) << "  Temporary data size: " << formatMem(usedMem) << "\n";
}

void PLT::getNodesToUpdate(UnorderedSet<TreeNode*>& nPositive, UnorderedSet<TreeNode*>& nNegative, const SparseVector& labels) {
    for (auto &l : labels) {
        auto ni = tree->leaves.find(l.index);
        if (ni == tree->leaves.end()) {
            Log(CERR) << "Encountered example with label " << l.index << " that does not exists in the tree\n";
            continue;
        }
        TreeNode* n = ni->second;
        nPositive.insert(n);
        while (n->parent) {
            n = n->parent;
            nPositive.insert(n);
        }
    }

    if (nPositive.empty()) {
        nNegative.insert(tree->root);
        return;
    }

    for(auto& n : nPositive) {
        for (const auto &child : n->children) {
            if (!nPositive.count(child))
                nNegative.insert(child);
        }
    }
}

void PLT::addNodesLabelsAndFeatures(std::vector<std::vector<Real>>& binLabels, std::vector<std::vector<Feature*>>& binFeatures,
                      UnorderedSet<TreeNode*>& nPositive, UnorderedSet<TreeNode*>& nNegative,
                      SparseVector& features) {
    Feature* featuresData = features.data();

    for (const auto& n : nPositive) {
        binLabels[n->index].push_back(1.0);
        binFeatures[n->index].push_back(featuresData);
    }

    for (const auto& n : nNegative) {
        binLabels[n->index].push_back(0.0);
        binFeatures[n->index].push_back(featuresData);
    }
}

std::vector<std::vector<Prediction>> PLT::predictBatch(SRMatrix& features, Args& args) {
    if (args.treeSearchType == exact) return Model::predictBatch(features, args);
    else if (args.treeSearchType == beam) return predictWithBeamSearch(features, args);
    else throw std::invalid_argument("Unknown tree search type");
}

std::vector<std::vector<Prediction>> PLT::predictWithBeamSearch(SRMatrix& features, Args& args){
    Log(CERR) << "Starting prediction in 1 thread ...\n";

    int rows = features.rows();
    int nodes = tree->nodes.size();

    std::vector<std::vector<Prediction>> prediction(rows);
    std::vector<std::vector<TreeNodeValue>> levelPredictions(rows);
    std::vector<std::vector<Prediction>> nodePredictions(nodes);

    // note: technically, these are queues, but we have well separated phases, where we first only
    // push elements, and after that, we only pop elements; thus, we can just use a vector, which is
    // a more efficient container.
    std::vector<TreeNode*> nextLevelQueue;
    std::vector<TreeNode*> levelQueue;
    nextLevelQueue.push_back(tree->root);
    for(int i = 0; i < rows; ++i) nodePredictions[tree->root->index].emplace_back(i, 1.0);
    AbstractVector* tmpW = new Vector(features.cols());
    AbstractVector* originalW = nullptr;

    int nCount = 0;
    while(!nextLevelQueue.empty()) {
        levelQueue.clear();
        // the swap in effect empties nextLevelQueue and assignes
        // the next level to `levelQueue` while keeping allocated
        // memory available for reuse.
        std::swap(nextLevelQueue, levelQueue);

        // Predict for level
        for(auto n: levelQueue) {
            printProgress(nCount++, nodes);
            int nIdx = n->index;

            if(!nodePredictions[nIdx].empty()){
                auto base = bases[nIdx];
                auto type = base->getType();
                if(type == sparse && args.beamSearchUnpack){
                    originalW = base->getW();
                    tmpW->add(*originalW);
                    base->setW(tmpW);
                }

                for(auto &e : nodePredictions[nIdx]){
                    int rIdx = e.label;
                    Real prob = bases[nIdx]->predictProbability(features[rIdx]) * e.value;
                    Real value = prob;

                    // Reweight score
                    if (!labelsWeights.empty()) value *= nodesWeights[nIdx].value + nodesBiases[nIdx].value;

                    if(n->label >= 0) prediction[rIdx].emplace_back(n->label, value); // Label prediction
                    if(!n->children.empty()) levelPredictions[rIdx].emplace_back(n, prob, value); // Internal node prediction
                }
                nodeEvaluationCount += nodePredictions[nIdx].size();
                nodePredictions[nIdx].clear();

                if(type == sparse && args.beamSearchUnpack){
                    tmpW->zero(*originalW);
                    base->setW(originalW);
                }
            }

            nextLevelQueue.insert(nextLevelQueue.end(), n->children.begin(), n->children.end());
        }

        // Keep top predictions and prepare next level
        for(int rIdx = 0; rIdx < rows; ++rIdx){
            auto &v = levelPredictions[rIdx];

            if(!thresholds.empty()){
                int j = 0;
                for(int i = 0; i < v.size(); ++i){
                    if(v[i].value > nodesThr[v[i].node->index].value)
                        v[j++] = v[i];
                }
                v.resize(j - 1);
            }
            else {
                std::sort(v.rbegin(), v.rend());

                if(args.threshold > 0){
                    int i = 0;
                    while (i < v.size() && v[i++].value > args.threshold);
                    v.resize(i - 1);
                }
                else v.resize(std::min(v.size(), (size_t)args.beamSearchWidth));
            }

            for(auto &nv : v)
                for(auto &c : nv.node->children)
                    nodePredictions[c->index].emplace_back(rIdx, nv.prob);
            v.clear();
        }
    }
    delete tmpW;

    for(int rIdx = 0; rIdx < rows; ++rIdx){
        auto &v = prediction[rIdx];
        std::sort(v.rbegin(), v.rend());
    }

    dataPointCount = rows;
    return prediction;
}

void PLT::predict(std::vector<Prediction>& prediction, SparseVector& features, Args& args) {
    int topK = args.topK;
    Real threshold = args.threshold;

    if(topK > 0) prediction.reserve(topK);
    TopKQueue<TreeNodeValue> nQueue(args.topK);


    // Set functions
    std::function<bool(TreeNode*, Real)> ifAddToQueue = [&] (TreeNode* node, Real prob) {
        return true;
    };

    if(args.threshold > 0)
        ifAddToQueue = [&] (TreeNode* node, Real prob) {
            return (prob >= threshold);
        };
    else if(thresholds.size())
        ifAddToQueue = [&] (TreeNode* node, Real prob) {
            return (prob >= nodesThr[node->index].value);
        };

    std::function<Real(TreeNode*, Real)> calculateValue = [&] (TreeNode* node, Real prob) {
        return prob;
    };

    if (!labelsWeights.empty())
        calculateValue = [&] (TreeNode* node, Real prob) {
            return prob * nodesWeights[node->index].value + nodesBiases[node->index].value;
        };

    // Predict for root
    Real rootProb = predictForNode(tree->root, features);
    addToQueue(ifAddToQueue, calculateValue, nQueue, tree->root, rootProb);
    ++nodeEvaluationCount;
    ++dataPointCount;

    Prediction p = predictNextLabel(ifAddToQueue, calculateValue, nQueue, features);
    while ((prediction.size() < topK || topK == 0) && p.label != -1) {
        prediction.push_back(p);
        p = predictNextLabel(ifAddToQueue, calculateValue, nQueue, features);
    }
}

Prediction PLT::predictNextLabel(
    std::function<bool(TreeNode*, Real)>& ifAddToQueue, std::function<Real(TreeNode*, Real)>& calculateValue,
    TopKQueue<TreeNodeValue>& nQueue, SparseVector& features) {
    while (!nQueue.empty()) {
        TreeNodeValue nVal = nQueue.top();
        nQueue.pop();

        if (!nVal.node->children.empty()) {
            for (const auto& child : nVal.node->children)
                addToQueue(ifAddToQueue, calculateValue, nQueue, child, nVal.prob * predictForNode(child, features));
            nodeEvaluationCount += nVal.node->children.size();
        }
        if (nVal.node->label >= 0) return {nVal.node->label, nVal.value};
    }

    return {-1, 0};
}

void PLT::calculateNodesLabels(){
    if(!tree) throw std::runtime_error("Tree is not constructed, load or build a tree first");

    if(tree->size() != nodesLabels.size()){
        nodesLabels.clear();
        nodesLabels.resize(tree->size());

        for (auto& l : tree->leaves) {
            TreeNode* n = l.second;
            while (n != nullptr) {
                nodesLabels[n->index].push_back(l.first);
                n = n->parent;
            }
        }
    }
}

void PLT::setNodeThreshold(TreeNode* n){
    if(!tree) throw std::runtime_error("Tree is not constructed, load or build a tree first");

    TreeNodeValueExt& nTh = nodesThr[n->index];
    nTh.value = std::numeric_limits<Real>::min();
    for (auto &l : nodesLabels[n->index]) {
        if (thresholds[l] < nTh.value) {
            nTh.value = thresholds[l];
            nTh.label = l;
        }
    }
}

void PLT::setNodeWeight(TreeNode* n){
    TreeNodeValueExt& nW = nodesWeights[n->index];
    nW.value = std::numeric_limits<Real>::min();
    for (auto &l : nodesLabels[n->index]) {
        if (labelsWeights[l] > nW.value) {
            nW.value = labelsWeights[l];
            nW.label = l;
        }
    }
}

void PLT::setNodeBias(TreeNode* n){
    TreeNodeValueExt& nB = nodesBiases[n->index];
    nB.value = std::numeric_limits<Real>::max();
    for (auto &l : nodesLabels[n->index]) {
        if (labelsBiases[l] > nB.value) {
            nB.value = labelsBiases[l];
            nB.label = l;
        }
    }
}

void PLT::setThresholds(std::vector<Real> th){
    if(!tree) throw std::runtime_error("Tree is not constructed, load or build a tree first");

    Model::setThresholds(std::move(th));
    calculateNodesLabels();
    if (tree->size() != nodesThr.size()) nodesThr.resize(tree->size());
    for (auto& n : tree->nodes) setNodeThreshold(n);
}

void PLT::setLabelsWeights(std::vector<Real> lw){
    if(!tree) throw std::runtime_error("Tree is not constructed, load or build a tree first");

    Model::setLabelsWeights(std::move(lw));
    calculateNodesLabels();
    if (tree->size() != nodesWeights.size()) nodesWeights.resize(tree->size());
    for (auto& n : tree->nodes) setNodeWeight(n);
}

void PLT::setLabelsBiases(std::vector<Real> lb){
    if(!tree) throw std::runtime_error("Tree is not constructed, load or build a tree first");

    Model::setLabelsBiases(lb);
    calculateNodesLabels();
    if (tree->size() != nodesBiases.size()) nodesBiases.resize(tree->size());
    for (auto& n : tree->nodes) setNodeBias(n);
}

void PLT::updateThresholds(UnorderedMap<int, Real> thToUpdate){
    for(auto& th : thToUpdate)
        thresholds[th.first] = th.second;

    for(auto& th : thToUpdate){
        TreeNode* n = tree->leaves[th.first];
        TreeNodeValueExt& nTh = nodesThr[n->index];
        while(n != tree->root){
            if(th.second < nTh.value){
                nTh.value = th.second;
                nTh.label = th.first;
            } else if (th.first == nTh.label && th.second > nTh.value){
                setNodeThreshold(n);
            }
            n = n->parent;
        }
    }
}

Real PLT::predictForLabel(Label label, SparseVector& features, Args& args) {
    auto fn = tree->leaves.find(label);
    if(fn == tree->leaves.end()) return 0;
    TreeNode* n = fn->second;
    Real value = bases[n->index]->predictProbability(features);
    while (n->parent) {
        n = n->parent;
        value *= predictForNode(n, features);
        ++nodeEvaluationCount;
    }

    if(!labelsWeights.empty())
        value *= labelsWeights[label];

    return value;
}

void PLT::preload(Args& args, std::string infile){
    tree = std::make_unique<LabelTree>();
    tree->loadFromFile(joinPath(infile, "tree.bin"));
    preloaded = true;
}

void PLT::load(Args& args, std::string infile) {
    Log(CERR) << "Loading " << name << " model ...\n";

    Log::updateGlobalIndent(2);
    preload(args, infile);
    bases = loadBases(joinPath(infile, "weights.bin"), args.resume, args.loadAs);

    assert(bases.size() == tree->nodes.size());
    m = tree->getNumberOfLeaves();

    loaded = true;
    Log::updateGlobalIndent(-2);
}

void PLT::printInfo() {
    Log(COUT) << name << " additional stats:"
              << "\n  Tree size: " << tree->nodes.size()
              << "\n  Tree depth: " << tree->getTreeDepth() << "\n";
    if(nodeUpdateCount > 0)
        Log(COUT) << "  Updated estimators / data point: " << static_cast<Real>(nodeUpdateCount) / dataPointCount << "\n";
    if(nodeEvaluationCount > 0)
        Log(COUT) << "  Evaluated estimators / data point: " << static_cast<Real>(nodeEvaluationCount) / dataPointCount << "\n";
}

void PLT::buildTree(SRMatrix& labels, SRMatrix& features, Args& args, const std::string& output){
    tree = std::make_unique<LabelTree>();
    tree->buildTreeStructure(labels, features, args);

    m = tree->getNumberOfLeaves();
    tree->saveToFile(joinPath(output, "tree.bin"));
    tree->saveTreeStructure(joinPath(output, "tree.txt"));
}

std::vector<std::vector<std::pair<int, Real>>> PLT::getNodesToUpdate(const SRMatrix& labels){
    if(!tree) throw std::runtime_error("Tree is not constructed, load or build a tree first");

    // Positive and negative nodes
    UnorderedSet<TreeNode*> nPositive;
    UnorderedSet<TreeNode*> nNegative;

    Log(CERR) << "Getting nodes to update ...\n";

    // Gather examples for each node
    int rows = labels.rows();
    std::vector<std::vector<std::pair<int, Real>>> nodesToUpdate(rows);

    for (int r = 0; r < rows; ++r) {
        printProgress(r, rows);

        nPositive.clear();
        nNegative.clear();

        getNodesToUpdate(nPositive, nNegative, labels[r]);
        nodesToUpdate[r].reserve(nPositive.size() + nNegative.size());
        for (const auto& n : nPositive) nodesToUpdate[r].emplace_back(n->index, 1.0);
        for (const auto& n : nNegative) nodesToUpdate[r].emplace_back(n->index, 0);
    }

    return nodesToUpdate;
}

std::vector<std::vector<std::pair<int, Real>>> PLT::getNodesUpdates(const SRMatrix& labels){
    if(!tree) throw std::runtime_error("Tree is not constructed, load or build a tree first");

    // Positive and negative nodes
    UnorderedSet<TreeNode*> nPositive;
    UnorderedSet<TreeNode*> nNegative;

    Log(CERR) << "Getting nodes to update ...\n";

    // Gather examples for each node
    int rows = labels.rows();
    std::vector<std::vector<std::pair<int, Real>>> nodesDataPoints(tree->size());

    for (int r = 0; r < rows; ++r) {
        printProgress(r, rows);

        nPositive.clear();
        nNegative.clear();

        getNodesToUpdate(nPositive, nNegative, labels[r]);
        for (const auto& n : nPositive) nodesDataPoints[n->index].emplace_back(r, 1.0);
        for (const auto& n : nNegative) nodesDataPoints[n->index].emplace_back(r, 0);
    }

    return nodesDataPoints;
}

void PLT::setTreeStructure(std::vector<std::tuple<int, int, int>> treeStructure, const std::string& output){
    if(tree == nullptr) tree = std::make_unique<LabelTree>();
    tree->setTreeStructure(std::move(treeStructure));

    m = tree->getNumberOfLeaves();
    tree->saveToFile(joinPath(output, "tree.bin"));
    tree->saveTreeStructure(joinPath(output, "tree.txt"));
}

std::vector<std::tuple<int, int, int>> PLT::getTreeStructure(){
    if(tree == nullptr) return std::vector<std::tuple<int, int, int>>();
    else return tree->getTreeStructure();
}

void BatchPLT::train(SRMatrix& labels, SRMatrix& features, Args& args, std::string output) {
    if(!tree) buildTree(labels, features, args, output);

    Log(CERR) << "Training tree ...\n";

    // Examples selected for each node
    std::vector<std::vector<Real>> binLabels(tree->size());
    std::vector<std::vector<Feature*>> binFeatures(tree->size());
    std::vector<std::vector<Real>> binWeights;

    if (type == hsm && args.pickOneLabelWeighting) binWeights.resize(tree->size());
    else binWeights.emplace_back(features.rows(), 1);

    assignDataPoints(binLabels, binFeatures, binWeights, labels, features, args);

    // Train bases
    std::vector<ProblemData> binProblemData;
    binProblemData.reserve(tree->size());
    if (type == hsm && args.pickOneLabelWeighting)
        for (int i = 0; i < tree->size(); ++i) binProblemData.emplace_back(binLabels[i], binFeatures[i], features.cols(), binWeights[i]);
    else
        for (int i = 0; i < tree->size(); ++i) binProblemData.emplace_back(binLabels[i], binFeatures[i], features.cols(), binWeights[0]);

    for (auto &pb: binProblemData) {
        pb.r = features.rows();
        pb.invPs = 1;
    }
    
    trainBases(joinPath(output, "weights.bin"), binProblemData, args);
}
