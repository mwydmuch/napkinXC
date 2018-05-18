/**
 * Copyright (c) 2018 by Marek Wydmuch, Kalina Jasinska, Robert Istvan Busa-Fekete
 * All rights reserved.
 */

#include <cassert>
#include <unordered_set>
#include <algorithm>
#include <vector>
#include <queue>
#include <list>
#include <chrono>
#include <random>
#include <cmath>
#include <climits>

#include "pltree.h"
#include "threads.h"
#include "knn.h"

PLTree::PLTree(){}

PLTree::~PLTree() {
    for(size_t i = 0; i < tree.size(); ++i)
        delete tree[i];
}

void PLTree::buildTreeStructure(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args){
    // Create a tree structure
    if (args.treeType == completeInOrder)
        buildCompleteTree(labels.cols(), false, args);
    else if (args.treeType == completeRandom)
        buildCompleteTree(labels.cols(), true, args);
    else if (args.treeType == balancedInOrder)
        buildBalancedTree(labels.cols(), false, args);
    else if (args.treeType == balancedRandom)
        buildBalancedTree(labels.cols(), true, args);
    else if (args.treeType == huffman)
        buildHuffmanTree(labels, args);
    else if (args.treeType == hierarchicalKMeans) {
        SRMatrix<Feature> labelsFeatures;
        computeLabelsFeaturesMatrix(labelsFeatures, labels, features);
        buildKMeansTree(labelsFeatures, args);
    }
    else if (args.treeType == leaveFreqBehind) {
        SRMatrix<Feature> labelsFeatures;
        std::vector<Frequency> labelsFreq;
        computeLabelsFeaturesMatrix(labelsFeatures, labels, features);
        computeLabelsFrequencies(labelsFreq, labels);
        buildLeaveFreqBehindTree(labelsFeatures, labelsFreq, args);
    }
    else if (args.treeType == kMeansHuffman) {
        SRMatrix<Feature> labelsFeatures;
        std::vector<Frequency> labelsFreq;
        computeLabelsFeaturesMatrix(labelsFeatures, labels, features);
        computeLabelsFrequencies(labelsFreq, labels);
        buildKMeansHuffmanTree(labelsFeatures, labelsFreq, labels, args);
    }
    else if(args.treeType == kMeansWithProjection)
        balancedKMeansWithRandomProjection(labels, features, args);
    else {
        std::cerr << "Unknown tree type\n";
        exit(0);
    }

    // Save structure
    saveTreeStructure(joinPath(args.model, "tree.txt"));
}

void PLTree::train(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args){
    rng.seed(args.seed);

    if(args.treeType == topDown){
        // Top down building and training
        trainTopDown(labels, features, args);
    } else {
        if (!args.tree.empty()) loadTreeStructure(args.tree);
        else buildTreeStructure(labels, features, args);

        // Train the tree structure
        trainTreeStructure(labels, features, args);
    }
}

Base* nodeTrainThread(int n, std::vector<double>& binLabels, std::vector<Feature*>& binFeatures, Args& args){
    Base* base = new Base();
    base->train(n, binLabels, binFeatures, args);
    return base;
}

void PLTree::trainTreeStructure(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args){
    std::cerr << "Training tree ...\n";

    // For stats
    long nCount = 0, yCount = 0;

    // Check data
    int rows = features.rows();
    assert(rows == labels.rows());
    assert(k >= labels.cols());

    // Check tree
    assert(k == treeLeaves.size());
    assert(t == tree.size());

    // Examples selected for each node
    std::vector<std::vector<double>> binLabels(t);
    std::vector<std::vector<Feature*>> binFeatures(t);

    // Positive and negative nodes
    std::unordered_set<TreeNode*> nPositive;
    std::unordered_set<TreeNode*> nNegative;

    std::cerr << "Assigning points to nodes ...\n";

    // Gather examples for each node
    for(int r = 0; r < rows; ++r){
        printProgress(r, rows);

        nPositive.clear();
        nNegative.clear();

        int rSize = labels.size(r);
        auto rLabels = labels.row(r);

        if (rSize > 0){
            for (int i = 0; i < rSize; ++i) {
                TreeNode *n = treeLeaves[rLabels[i]];
                //if(!n->parent->kNNNode) nPositive.insert(n); //kNN base classifiers will be left empty
                nPositive.insert(n);
                while (n->parent) {
                    n = n->parent;
                    //if(!n->parent->kNNNode) nPositive.insert(n);
                    nPositive.insert(n);
                }
            }

            std::queue<TreeNode*> nQueue; // Nodes queue
            nQueue.push(treeRoot); // Push root

            while(!nQueue.empty()) {
                TreeNode* n = nQueue.front(); // Current node
                nQueue.pop();

                for(const auto& child : n->children) {
                    if (nPositive.count(child)) nQueue.push(child);
                    else nNegative.insert(child);
                }
            }
        } else nNegative.insert(treeRoot);

        for (const auto& n : nPositive){
            binLabels[n->index].push_back(1.0);
            binFeatures[n->index].push_back(features.row(r));
        }

        for (const auto& n : nNegative){
            binLabels[n->index].push_back(0.0);
            binFeatures[n->index].push_back(features.row(r));
        }

        nCount += nPositive.size() + nNegative.size();
        yCount += rSize;
    }

    std::cerr << "Starting training in " << args.threads << " threads ...\n";

    std::ofstream weightsOut(joinPath(args.model, "weights.bin"));
    if(args.threads > 1){
        // Run learning in parallel
        ThreadPool tPool(args.threads);
        std::vector<std::future<Base*>> results;

        for(const auto& n : tree)
            results.emplace_back(tPool.enqueue(nodeTrainThread, features.cols(), std::ref(binLabels[n->index]),
                                               std::ref(binFeatures[n->index]), std::ref(args)));

        // Saving in the main thread
        for(int i = 0; i < results.size(); ++i) {
            printProgress(i, results.size());
            Base* base = results[i].get();
            base->save(weightsOut, args);
            delete base;
        }
    } else {
        for(int i = 0; i < tree.size(); ++i){
            printProgress(i, tree.size());
            Base base;
            base.train(features.cols(), binLabels[tree[i]->index], binFeatures[tree[i]->index], args);
            base.save(weightsOut, args);
        }
    }
    weightsOut.close();

    std::cerr << "  Points count: " << rows
              << "\n  Nodes per point: " << static_cast<double>(nCount) / rows
              << "\n  Labels per point: " << static_cast<double>(yCount) / rows
              << "\n";

    if(args.kNN) {
        std::vector<std::vector<Example>> labelsExamples;
        computeLabelsExamples(labelsExamples, labels);

        std::cerr << "Starting build kNN classifier ...\n";

        int kNNNodes = 0, kNNChildren = 0;
        std::ofstream kNNOut(joinPath(args.model, "knn.bin"));
        for (const auto &n : tree) {
            if (n->kNNNode) {
                KNN knn;
                knn.build(n->children, labelsExamples);
                knn.save(kNNOut);

                ++kNNNodes;
                kNNChildren += n->children.size();
            }
        }
        kNNOut.close();
        std::cerr << "  K-NN nodes: " << kNNNodes << ", K-NN children: " << kNNChildren << "\n";

        std::cerr << "Saving training data ...\n";
        std::ofstream trainOut(joinPath(args.model, "train.bin"));
        labels.save(trainOut);
        features.save(trainOut);
        trainOut.close();
    }

    // Save tree
    save(joinPath(args.model, "tree.bin"));

    // Save args
    args.save(joinPath(args.model, "args.bin"));

    std::cerr << "All done\n";
}

void PLTree::predict(std::vector<TreeNodeValue>& prediction, Feature* features, std::vector<Base*>& bases, std::vector<KNN*>& kNNs, Args &args){
    std::priority_queue<TreeNodeValue> nQueue;

    // Note: loss prediction gets worse results for tree with higher arity then 2

    double val = bases[treeRoot->index]->predictProbability(features);
    //double val = -bases[treeRoot->index]->predictLoss(features);
    nQueue.push({treeRoot, val});

    while (!nQueue.empty()) {
        TreeNodeValue nVal = nQueue.top(); // Current node
        nQueue.pop();

        if(nVal.node->label >= 0){
            prediction.push_back({nVal.node, nVal.value}); // When using probability
            //prediction.push_back({nVal.node, exp(nVal.value)}); // When using loss
            if (prediction.size() >= args.topK)
                break;
        } else {
            if(nVal.node->kNNNode && args.kNN){ // KNN supports only probabilities
                TreeNode* n = nVal.node;
                std::vector<Feature> result;
                kNNs[nVal.node->index]->predict(features, args.kNN, result);
                for(const auto& r : result){
                    val = nVal.value * r.value;
                    nQueue.push({tree[r.index], val});
                }
            }
            for(const auto& child : nVal.node->children){
                val = nVal.value * bases[child->index]->predictProbability(features); // When using probability
                //val = nVal.value - bases[child->index]->predictLoss(features); // When using loss
                nQueue.push({child, val});
            }
        }
    }
}

std::mutex testMutex;
int pointTestThread(PLTree* tree, Label* labels, Feature* features, std::vector<Base*>& bases, std::vector<KNN*>& kNNs,
    Args& args, std::vector<int>& correctAt, std::vector<std::unordered_set<int>>& coveredAt){

    std::vector<TreeNodeValue> prediction;
    tree->predict(prediction, features, bases, kNNs, args);

    testMutex.lock();
    for (int i = 0; i < args.topK; ++i){
        int l = -1;
        while(labels[++l] > -1)
            if (prediction[i].node->label == labels[l]){
                ++correctAt[i];
                coveredAt[i].insert(prediction[i].node->label);
                break;
            }
    }
    testMutex.unlock();

    return 0;
}

int batchTestThread(PLTree* tree, SRMatrix<Label>& labels, SRMatrix<Feature>& features,
    std::vector<Base*>& bases, std::vector<KNN*>& kNNs, Args& args, int startRow, int stopRow,
    std::vector<int>& correctAt, std::vector<std::unordered_set<int>>& coveredAt){

    std::vector<int> localCorrectAt (args.topK);
    for(int r = startRow; r < stopRow; ++r){
        std::vector<TreeNodeValue> prediction;
        tree->predict(prediction, features.row(r), bases, kNNs, args);

        for (int i = 0; i < args.topK; ++i)
            for (int j = 0; j < labels.size(r); ++j)
                if (prediction[i].node->label == labels.row(r)[j]){
                    ++localCorrectAt[i];
                    coveredAt[i].insert(prediction[i].node->label);
                    break;
                }
    }

    testMutex.lock();
    for (int i = 0; i < args.topK; ++i)
        correctAt[i] += localCorrectAt[i];
    testMutex.unlock();

    return 0;
}

void PLTree::test(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args) {
    std::cerr << "Loading base classifiers ...\n";
    std::vector<Base*> bases;
    std::ifstream weightsIn(joinPath(args.model, "weights.bin"));
    for(int i = 0; i < t; ++i) {
        printProgress(i, t);
        bases.emplace_back(new Base());
        bases.back()->load(weightsIn, args);
    }
    weightsIn.close();

    SRMatrix<Label> trainLabels;
    SRMatrix<Feature> trainFeatures;
    std::vector<KNN*> kNNs;

    if(args.kNN) {
        std::cerr << "Loading kNN classifiers ...\n";
        int kNNNodes = 0, kNNChildren = 0;
        std::ifstream kNNIn(joinPath(args.model, "knn.bin"));
        for (int i = 0; i < t; ++i) {
            printProgress(i, t);
            if (tree[i]->kNNNode) {
                kNNs.emplace_back(new KNN(&trainLabels, &trainFeatures));
                kNNs.back()->load(kNNIn);
                ++kNNNodes;
                kNNChildren += tree[i]->children.size();
            } else kNNs.push_back(nullptr);
        }
        kNNIn.close();
        std::cerr << "  K-NN nodes: " << kNNNodes << ", K-NN children: " << kNNChildren << "\n";

        std::cerr << "Loading training data ...\n";
        std::ifstream trainIn(joinPath(args.model, "train.bin"));
        trainLabels.load(trainIn);
        trainFeatures.load(trainIn);
        trainIn.close();
    }

    std::cerr << "Starting testing ...\n";

    std::vector<int> correctAt(args.topK);
    std::vector<std::unordered_set<int>> coveredAt(args.topK);
    int rows = features.rows();
    assert(rows == labels.rows());

    if(args.threads > 1){
        // Run prediction in parallel

        // Pool
        ThreadPool tPool(args.threads);
        std::vector<std::future<int>> results;

        for(int r = 0; r < rows; ++r)
            results.emplace_back(tPool.enqueue(pointTestThread, this, labels.row(r), features.row(r), std::ref(bases),
                                               std::ref(kNNs), std::ref(args), std::ref(correctAt), std::ref(coveredAt)));

        for(int i = 0; i < results.size(); ++i) {
            printProgress(i, results.size());
            results[i].get();
        }

        // Batches
        /*
        ThreadSet tSet;
        int tRows = ceil(static_cast<double>(rows)/args.threads);
        for(int t = 0; t < args.threads; ++t)
            tSet.add(batchTestThread, this, std::ref(labels), std::ref(features), std::ref(bases),
                args.topK, t * tRows, std::min((t + 1) * tRows, labels.rows()), std::ref(precisionAt));
        tSet.joinAll();
        */

    } else {
        std::vector<TreeNodeValue> prediction;
        for(int r = 0; r < rows; ++r){
            prediction.clear();
            predict(prediction, features.data()[r], bases, kNNs, args);

            for (int i = 0; i < args.topK; ++i)
                for (int j = 0; j < labels.sizes()[r]; ++j)
                    if (prediction[i].node->label == labels.data()[r][j]){
                        ++correctAt[i];
                        coveredAt[i].insert(prediction[i].node->label);
                        break;
                    }
            printProgress(r, rows);
        }
    }

    double precisionAt = 0, coverageAt;
    for (int i = 0; i < args.topK; ++i) {
        if(i > 0) for(const auto& l : coveredAt[i - 1]) coveredAt[i].insert(l);
        precisionAt += correctAt[i];
        coverageAt = coveredAt[i].size();
        std::cerr << "P@" << i + 1 << ": " << precisionAt / (rows * (i + 1)) << ", C@" << i + 1 << ": " << coverageAt / labels.cols() << "\n";
    }

    for(auto base : bases) delete base;
    for(auto knn : kNNs) delete knn;

    std::cerr << "All done\n";
}

TreeNodePartition treeNodeKMeansThread(TreeNodePartition nPart, SRMatrix<Feature>& labelsFeatures, Args& args, int seed){
    kMeans(nPart.partition, labelsFeatures, args.arity, args.kMeansEps, args.kMeansBalanced, seed);
    return nPart;
}

std::vector<Assignation>* partitionKMeansThread(std::vector<Assignation>* partition, SRMatrix<Feature>& labelsFeatures, Args& args, int seed){
    kMeans(partition, labelsFeatures, args.arity, args.kMeansEps, args.kMeansBalanced, seed);
    return partition;
}

void PLTree::buildLeaveFreqBehindTree(SRMatrix<Feature>& labelsFeatures, std::vector<Frequency>& labelsFreq, Args& args){
    std::cerr << "\"Leave Freq Behind\" tree in " << args.threads << " threads ...\n";

    treeRoot = createTreeNode();
    k = labelsFeatures.rows();

    std::uniform_int_distribution<int> kMeansSeeder(0, INT_MAX);

    int kNNLabels = 0, kNNNodes = 0;

    auto partition = new std::vector<Assignation>(k);
    for(int i = 0; i < k; ++i) (*partition)[i].index = i;

    if(args.threads > 1){
        // Run clustering in parallel
        ThreadPool tPool(args.threads);
        std::vector<std::future<TreeNodePartition>> results;

        TreeNodePartition rootPart = {treeRoot, partition};
        results.emplace_back(tPool.enqueue(treeNodeKMeansThread, rootPart, std::ref(labelsFeatures),
                                           std::ref(args), kMeansSeeder(rng)));

        for(int r = 0; r < results.size(); ++r) {
            // Enqueuing new clustering tasks in the main thread ensures determinism
            TreeNodePartition nPart = results[r].get();

            // This needs to be done this way in case of imbalance K-Means
            auto partitions = new std::vector<Assignation>* [args.arity];
            for (int i = 0; i < args.arity; ++i) partitions[i] = new std::vector<Assignation>();
            for (auto a : *nPart.partition) partitions[a.value]->push_back({a.index, 0});

            for (int i = 0; i < args.arity; ++i) {
                TreeNode *n = createTreeNode(nPart.node);

                if(partitions[i]->size() <= args.maxLeaves) {
                    /*
                    double avg = 0;
                    for (const auto& a : *partitions[i]){
                        avg += labelsFreq[a.index].value;
                    }

                    avg /= partitions[i]->size();
                     */
                    double avg = args.kNNMaxFreq;
                    TreeNode* kNNN = nullptr;
                    for (const auto& a : *partitions[i]){
                        if(labelsFreq[a.index].value > avg) createTreeNode(n, a.index);
                        else{
                            if(kNNN == nullptr){
                                kNNN = createTreeNode(n);
                                kNNN->kNNNode = true;
                                ++kNNNodes;
                            }
                            createTreeNode(kNNN, a.index);
                            ++kNNLabels;
                        }
                    }

                    delete partitions[i];
                } else {
                    TreeNodePartition childPart = {n, partitions[i]};
                    results.emplace_back(tPool.enqueue(treeNodeKMeansThread, childPart, std::ref(labelsFeatures),
                                                       std::ref(args), kMeansSeeder(rng)));
                }
            }

            delete nPart.partition;
        }
    } else {
        std::queue<TreeNodePartition> nQueue;
        nQueue.push({treeRoot, partition});

        while (!nQueue.empty()) {
            TreeNodePartition nPart = nQueue.front(); // Current node
            nQueue.pop();

            if (nPart.partition->size() > args.maxLeaves) {
                kMeans(nPart.partition, labelsFeatures, args.arity, args.kMeansEps, args.kMeansBalanced, kMeansSeeder(rng));
                auto partitions = new std::vector<Assignation>* [args.arity];
                for (int i = 0; i < args.arity; ++i) partitions[i] = new std::vector<Assignation>();
                for (auto a : *nPart.partition) partitions[a.value]->push_back({a.index, 0});

                // Create children
                for (int i = 0; i < args.arity; ++i) {
                    TreeNode *n = createTreeNode(nPart.node);
                    nQueue.push({n, partitions[i]});
                }
            } else
                for (const auto& a : *nPart.partition) createTreeNode(nPart.node, a.index);

            delete nPart.partition;
        }
    }

    t = tree.size();
    assert(k == treeLeaves.size());
    std::cerr << "  Nodes: " << tree.size() << ", K-NN nodes: " << kNNNodes << ", leaves: " << treeLeaves.size() << ", K-NN leaves: " << kNNLabels << "\n";

}

// TODO: Finish this method
void PLTree::buildKMeansHuffmanTree(SRMatrix<Feature>& labelsFeatures, std::vector<Frequency>& labelsFreq, SRMatrix<Label>& labels, Args& args){
    std::cerr << "\"K-Means X Huffman\" tree in " << args.threads << " threads ...\n";

    treeRoot = createTreeNode();
    k = labelsFeatures.rows();

    std::uniform_int_distribution<int> kMeansSeeder(0, INT_MAX);

    std::vector<std::vector<Assignation>*> finalPartition;
    std::vector<Assignation>* partition = new std::vector<Assignation>(k);
    for(int i = 0; i < k; ++i) (*partition)[i].index = i;

    if(args.threads > 1){
        // Run clustering in parallel
        ThreadPool tPool(args.threads);
        std::vector<std::future<std::vector<Assignation>*>> results;
        results.emplace_back(tPool.enqueue(partitionKMeansThread, partition, std::ref(labelsFeatures),
                                           std::ref(args), kMeansSeeder(rng)));

        for(int r = 0; r < results.size(); ++r) {
            // Enqueuing new clustering tasks in the main thread ensures determinism
            partition = results[r].get();

            // This needs to be done this way in case of imbalanced K-Means
            auto partitions = new std::vector<Assignation>* [args.arity];
            for (int i = 0; i < args.arity; ++i) partitions[i] = new std::vector<Assignation>();
            for (auto a : *partition) partitions[a.value]->push_back({a.index, 0});

            for (int i = 0; i < args.arity; ++i) {
                if(partitions[i]->size() <= args.maxLeaves) finalPartition.push_back(partitions[i]);
                else results.emplace_back(tPool.enqueue(partitionKMeansThread, partitions[i], std::ref(labelsFeatures),
                                                       std::ref(args), kMeansSeeder(rng)));
            }

            delete partition;
        }
    } else {
        std::queue<std::vector<Assignation>*> nQueue;
        nQueue.push(partition);

        while (!nQueue.empty()) {
            partition = nQueue.front();
            nQueue.pop();

            if (partition->size() > args.maxLeaves) {
                kMeans(partition, labelsFeatures, args.arity, args.kMeansEps, args.kMeansBalanced, kMeansSeeder(rng));
                auto partitions = new std::vector<Assignation>* [args.arity];
                for (int i = 0; i < args.arity; ++i) partitions[i] = new std::vector<Assignation>();
                for (auto a : *partition) partitions[a.value]->push_back({a.index, 0});

                for (int i = 0; i < args.arity; ++i) nQueue.push(partitions[i]);
                delete partition;
            } else finalPartition.push_back(partition);
        }
    }

    // Calculate clusters probability
    std::priority_queue<TreeNodeFrequency> nodeFreq;
    for(int c = 0; c < finalPartition.size(); ++c){
        std::unordered_set<int> partitionsLabels;
        for(const auto& a : *finalPartition[c])
            partitionsLabels.insert(a.index);

        int freq = 0;
        int rows = labels.rows();
        for(int r = 0; r < rows; ++r){

        }
    }

    // Build Huffman tree on top of clusters
    t = tree.size();
    assert(k == treeLeaves.size());
    std::cerr << "  Nodes: " << tree.size() << ", leaves: " << treeLeaves.size() << "\n";
}


void PLTree::buildKMeansTree(SRMatrix<Feature>& labelsFeatures, Args& args){
    std::cerr << "Hierarchical K-Means clustering in " << args.threads << " threads ...\n";

    treeRoot = createTreeNode();
    k = labelsFeatures.rows();

    std::uniform_int_distribution<int> kMeansSeeder(0, INT_MAX);

    auto partition = new std::vector<Assignation>(k);
    for(int i = 0; i < k; ++i) (*partition)[i].index = i;

    if(args.threads > 1){
        // Run clustering in parallel
        ThreadPool tPool(args.threads);
        std::vector<std::future<TreeNodePartition>> results;

        TreeNodePartition rootPart = {treeRoot, partition};
        results.emplace_back(tPool.enqueue(treeNodeKMeansThread, rootPart, std::ref(labelsFeatures),
                                           std::ref(args), kMeansSeeder(rng)));

        for(int r = 0; r < results.size(); ++r) {
            // Enqueuing new clustering tasks in the main thread ensures determinism
            TreeNodePartition nPart = results[r].get();

            // This needs to be done this way in case of imbalanced K-Means
            auto partitions = new std::vector<Assignation>* [args.arity];
            for (int i = 0; i < args.arity; ++i) partitions[i] = new std::vector<Assignation>();
            for (auto a : *nPart.partition) partitions[a.value]->push_back({a.index, 0});

            for (int i = 0; i < args.arity; ++i) {
                if(partitions[i]->empty()) continue;
                else if(partitions[i]->size() == 1){
                    createTreeNode(nPart.node, partitions[i]->front().index);
                    delete partitions[i];
                    continue;
                }

                TreeNode *n = createTreeNode(nPart.node);

                if(partitions[i]->size() <= args.maxLeaves) {
                    //n->kNNNode = true;
                    for (const auto& a : *partitions[i]) createTreeNode(n, a.index);
                    delete partitions[i];
                } else {
                    TreeNodePartition childPart = {n, partitions[i]};
                    results.emplace_back(tPool.enqueue(treeNodeKMeansThread, childPart, std::ref(labelsFeatures),
                                                       std::ref(args), kMeansSeeder(rng)));
                }
            }

            delete nPart.partition;
        }
    } else {
        std::queue<TreeNodePartition> nQueue;
        nQueue.push({treeRoot, partition});

        while (!nQueue.empty()) {
            TreeNodePartition nPart = nQueue.front(); // Current node
            nQueue.pop();

            if (nPart.partition->size() > args.maxLeaves) {
                kMeans(nPart.partition, labelsFeatures, args.arity, args.kMeansEps, args.kMeansBalanced, kMeansSeeder(rng));
                auto partitions = new std::vector<Assignation>* [args.arity];
                for (int i = 0; i < args.arity; ++i) partitions[i] = new std::vector<Assignation>();
                for (auto a : *nPart.partition) partitions[a.value]->push_back({a.index, 0});

                // Create children
                for (int i = 0; i < args.arity; ++i) {
                    TreeNode *n = createTreeNode(nPart.node);
                    nQueue.push({n, partitions[i]});
                }
            } else
                for (const auto& a : *nPart.partition) createTreeNode(nPart.node, a.index);

            delete nPart.partition;
        }
    }

    t = tree.size();
    assert(k == treeLeaves.size());
    std::cerr << "  Nodes: " << tree.size() << ", leaves: " << treeLeaves.size() << "\n";
}

void PLTree::balancedKMeansWithRandomProjection(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args &args) {

    int k = labels.cols();
    int n = features.rows();
    int dim = features.cols();

    std::cerr << "  Compute label to indices ...\n";
    std::vector<std::vector<int>> labelToIndices(k);
    for (int r = 0; r < n; ++r) {
        int rSize = labels.size(r);
        auto rLabels = labels.row(r);
        for (int i = 0; i < rSize; ++i) labelToIndices[rLabels[i]].push_back(r);
    }

    // Apply random projection
    std::vector<std::vector<double>> randomMatrix;
    generateRandomProjection(randomMatrix, args.projectDim, dim);
    SRMatrix<Feature> labelsFeatures(k, args.projectDim);
    projectLabelsRepresentation(labelsFeatures, randomMatrix, labelToIndices, features, args);

    // Build tree using hierarchical K-Means
    buildKMeansTree(labelsFeatures, args);
}

void PLTree::projectLabelsRepresentation(SRMatrix<Feature>& labelsFeatures, std::vector<std::vector<double>>& randomMatrix,
                                        std::vector<std::vector<int>>& labelToIndices, SRMatrix<Feature>& features, Args &args){

    int labels = labelToIndices.size();

    double scale = 1.0/sqrt(static_cast<double>(features.cols()));
    std::normal_distribution<double> distribution(0.0, scale);

    std::cerr << "  Compute projected values ...\n";
    for(int i=0; i < labels; i++ ){
        printProgress(i, labels);
        int currentLabel = i;
        auto labelVector = labelsFeatures.row(i);

        if (labelToIndices[currentLabel].size() > 0) {
            for (int j = 0; j < labelToIndices[currentLabel].size(); j++) {

                int currentDataPoint = labelToIndices[currentLabel][j];
                auto rFeatures = features.row(currentDataPoint);
                int rFeaturesSize = features.size(currentDataPoint);

                for (int l = 0; l < args.projectDim; l++)
                    for (int k = 0; k < rFeaturesSize; k++)
                        labelVector[l].value += rFeatures[k].value * randomMatrix[l][rFeatures[k].index];
                for (int l = 0; l < args.projectDim; l++)
                    labelVector[l].value /= labelToIndices[currentLabel].size();

                // Print row from labels' features matrix
                /*
                for (int l = 0; l < args.projectDim; l++) {
                    std::cout << labelsFeatures.data()[currentLabel][l].value << " ";
                std::cout << "\n";
                */
            }
        } else {
            for (int l = 0; l < args.projectDim; l++)
                labelVector[l].value = distribution(rng);
        }
    }
}

void PLTree::generateRandomProjection(std::vector<std::vector<double>>& randomMatrix, int projectDim, int dim){
    double scale = 1.0 / std::sqrt(static_cast<double>(dim));
    std::normal_distribution<double> distribution(0.0, scale);

    randomMatrix.resize(projectDim);
    for(int i = 0; i < projectDim; ++i) { // args.projectDim
        randomMatrix[i].resize(dim);
        for (int j = 0; j < dim; ++j) // dim
            randomMatrix[i][j] = distribution(rng);
    }
}


void PLTree::buildHuffmanTree(SRMatrix<Label>& labels, Args &args){
    std::cout << "Building Huffman PLTree ...\n";

    k = labels.cols();

    std::vector<Frequency> labelsFreq;
    computeLabelsFrequencies(labelsFreq, labels);

    std::priority_queue<TreeNodeFrequency> freqQueue;
    for(int i = 0; i < k; i++) {
        TreeNode* n = createTreeNode(nullptr, i);
        freqQueue.push({n, labelsFreq[i].value});
    }
    
    while(!freqQueue.empty()){
        std::vector<TreeNodeFrequency> toMerge;
        for(int a = 0; a < args.arity; ++a){
            toMerge.push_back(freqQueue.top());
            freqQueue.pop();
            if (freqQueue.empty()) break;
        }

        TreeNode* parent = createTreeNode();
        int aggregatedFreq = 0;
        for(TreeNodeFrequency& e : toMerge){
            e.node->parent = parent;
            parent->children.push_back(e.node);
            aggregatedFreq += e.frequency;
        }

        tree.push_back(parent);

        if (freqQueue.empty()) treeRoot = parent;
        freqQueue.push({parent, aggregatedFreq});
    }

    t = tree.size();  // size of the tree
    std::cout << "  Nodes: " << tree.size() << ", leaves: " << treeLeaves.size() << ", arity: " << args.arity << "\n";
}

void PLTree::buildBalancedTree(int labelCount, bool randomizeOrder, Args &args) {
    std::cerr << "Building balanced PLTree ...\n";

    treeRoot = createTreeNode();
    k = labelCount;

    auto partition = new std::vector<Assignation>(k);
    for(int i = 0; i < k; ++i) (*partition)[i].index = i;

    if (randomizeOrder) std::shuffle(partition->begin(), partition->end(), rng);

    std::queue<TreeNodePartition> nQueue;
    nQueue.push({treeRoot, partition});

    while (!nQueue.empty()) {
        TreeNodePartition nPart = nQueue.front(); // Current node
        nQueue.pop();
        if (nPart.partition->size() > args.maxLeaves) {
            auto partitions = new std::vector<Assignation>* [args.arity];
            for(int i = 0; i < args.arity; ++i) partitions[i] = new std::vector<Assignation>();

            int maxPartitionSize = nPart.partition->size() / args.arity;
            int maxWithOneMore = nPart.partition->size() % args.arity;
            int nextPartition = maxPartitionSize + (maxWithOneMore > 0 ? 1 : 0);
            int partitionNumber = 0;

            for (int i = 0; i < nPart.partition->size(); ++i) {
                if (i == nextPartition) {
                    ++partitionNumber;
                    --maxWithOneMore;
                    nextPartition += maxPartitionSize + (maxWithOneMore > 0 ? 1 : 0);
                    assert(partitionNumber < args.arity);
                }
                auto a = nPart.partition->at(i);
                partitions[partitionNumber]->push_back({a.index, 0});
            }
            assert(nextPartition == nPart.partition->size());

            // Create children
            for (int i = 0; i < args.arity; ++i) {
                TreeNode *n = createTreeNode(nPart.node);
                nQueue.push({n, partitions[i]});
            }
        } else
            for (const auto& a : *nPart.partition) createTreeNode(nPart.node, a.index);

        delete nPart.partition;
    }

    t = tree.size();
    assert(k == treeLeaves.size());
    std::cerr << "  Nodes: " << tree.size() << ", leaves: " << treeLeaves.size() << "\n";
}

void PLTree::buildCompleteTree(int labelCount, bool randomizeOrder, Args &args) {
    std::cerr << "Building complete PLTree ...\n";

    k = labelCount;
    t = static_cast<int>(ceil(static_cast<double>(args.arity * k - 1) / (args.arity - 1)));

    int ti = t - k;

    std::vector<int> labelsOrder;
    if (randomizeOrder){
        labelsOrder.resize(k);
        for (auto i = 0; i < k; ++i) labelsOrder[i] = i;
        std::shuffle(labelsOrder.begin(), labelsOrder.end(), rng);
    }

    treeRoot = createTreeNode();
    for(size_t i = 1; i < t; ++i){
        int label = -1;
        TreeNode *parent = nullptr;

        if(i >= ti){
            if(randomizeOrder) label = labelsOrder[i - ti];
            else label = i - ti;
        }

        parent = tree[static_cast<int>(floor(static_cast<double>(i - 1) / args.arity))];
        createTreeNode(parent, label);
    }

    std::cerr << "  Nodes: " << tree.size() << ", leaves: " << treeLeaves.size() << ", arity: " << args.arity << "\n";
}

void PLTree::loadTreeStructure(std::string file){
    std::cerr << "Loading PLTree structure from: " << file << "...\n";

    std::ifstream in(file);
    in >> k >> t;

    if(k >= t) throw "Specified number of labels is higher then specified number of nodes!\n";

    treeRoot = createTreeNode();
    for (int i = 1; i < t; ++i) createTreeNode();

    for (auto i = 0; i < t - 1; ++i) {
        int parent, child, label;
        in >> parent >> child >> label;

        if(child >= t) throw "Node index is higher then specified number of nodes!";
        if(parent >= t) throw "Parent index is higher then specified number of nodes!";
        if(label >= k) throw "Label index is higher then specified number of labels!";

        if(parent == -1){
            treeRoot = tree[child];
            --i;
            continue;
        }

        TreeNode *parentN = tree[parent];
        TreeNode *childN = tree[child];
        parentN->children.push_back(childN);
        childN->parent = parentN;

        if(label >= 0){
            assert(treeLeaves.count(label) == 0);
            assert(label < k);
            childN->label = label;
            treeLeaves[childN->label] = childN;
        }
    }
    in.close();

    // Additional validation of a tree
    for(const auto& n : tree) {
        if(n->parent == nullptr && n != treeRoot) throw "A node without parent, that is not a tree root exists!";
        if(n->children.size() == 0 && n->label < 0) throw "An internal node without children exists!";
    }

    assert(tree.size() == t);
    assert(treeLeaves.size() == k);
    std::cout << "  Nodes: " << tree.size() << ", leaves: " << treeLeaves.size() << "\n";
}

void PLTree::saveTreeStructure(std::string file) {
    std::cerr << "Saving PLTree structure to: " << file << "...\n";

    std::ofstream out(file);
    out << t << " " << k << "\n";
    for (auto &n : tree) {
        if (n->parent == nullptr) out << -1;
        else out << n->parent->index;

        out << " " << n->index << " ";

        if (n->label >= 0) out << n->label;
        else out << -1;

        out << "\n";
    }
    out.close();
}

TreeNode* PLTree::createTreeNode(TreeNode* parent, int label){
    TreeNode* n = new TreeNode();
    n->index = tree.size();
    n->label = label;
    n->parent = parent;
    n->kNNNode = false;
    if(label >= 0) treeLeaves[n->label] = n;
    if(parent != nullptr) parent->children.push_back(n);
    tree.push_back(n);

    return n;
}

void PLTree::save(std::string outfile){
    std::ofstream out(outfile);
    save(out);
    out.close();
}

void PLTree::save(std::ostream& out){
    std::cerr << "Saving PLTree model ...\n";

    out.write((char*) &k, sizeof(k));

    t = tree.size();
    out.write((char*) &t, sizeof(t));
    for(size_t i = 0; i < t; ++i) {
        TreeNode *n = tree[i];
        out.write((char*) &n->index, sizeof(n->index));
        out.write((char*) &n->label, sizeof(n->label));
        out.write((char*) &n->kNNNode, sizeof(n->kNNNode));
    }

    int rootN = treeRoot->index;
    out.write((char*) &rootN, sizeof(rootN));

    for(size_t i = 0; i < t; ++i) {
        TreeNode *n = tree[i];

        int parentN;
        if(n->parent) parentN = n->parent->index;
        else parentN = -1;

        out.write((char*) &parentN, sizeof(parentN));
    }
}

void PLTree::load(std::string infile){
    std::ifstream in(infile);
    load(in);
    in.close();
}

void PLTree::load(std::istream& in){
    std::cerr << "Loading PLTree model ...\n";

    in.read((char*) &k, sizeof(k));
    in.read((char*) &t, sizeof(t));
    for(size_t i = 0; i < t; ++i) {
        TreeNode *n = new TreeNode();
        in.read((char*) &n->index, sizeof(n->index));
        in.read((char*) &n->label, sizeof(n->label));
        in.read((char*) &n->kNNNode, sizeof(n->kNNNode));

        tree.push_back(n);
        if (n->label >= 0) treeLeaves[n->label] = n;
    }

    int rootN;
    in.read((char*) &rootN, sizeof(rootN));
    treeRoot = tree[rootN];

    for(size_t i = 0; i < t; ++i) {
        TreeNode *n = tree[i];

        int parentN;
        in.read((char*) &parentN, sizeof(parentN));
        if(parentN >= 0) {
            tree[parentN]->children.push_back(n);
            n->parent = tree[parentN];
        }
    }

    std::cerr << "  Nodes: " << tree.size() << ", leaves: " << treeLeaves.size() << "\n";
}

void PLTree::printTree(TreeNode *root){
    if(root == nullptr) root = treeRoot;

    std::unordered_set<TreeNode*> nSet;
    std::queue<TreeNode*> nQueue;
    nQueue.push(root);
    nSet.insert(root);
    int depth = 0;

    while(!nQueue.empty()){
        TreeNode* n = nQueue.front();
        nQueue.pop();

        if(nSet.count(n->parent)){
            nSet.clear();
            std::cerr << "\nDepth " << ++depth << ":";
        }

        nSet.insert(n);
        std::cerr << " " << n->index;
        if(n->parent) std::cerr << "(" << n->parent->index << ")";
        for(auto c : n->children) nQueue.push(c);
    }

    std::cerr << "\n";
}
