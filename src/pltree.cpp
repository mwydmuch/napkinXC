/**
 * Copyright (c) 2018 by Marek Wydmuch, Kalina Jasińska, Robert Istvan Busa-Fekete
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
#include "utils.h"
#include "threads.h"

PLTree::PLTree(){}

PLTree::~PLTree() {
    for(size_t i = 0; i < tree.size(); ++i){
        delete tree[i];
    }
}

Base* nodeTrainThread(int i, int n, std::vector<double>& binLabels, std::vector<Feature*>& binFeatures, Args& args){
    Base* base = new Base();
    base->train(n, binLabels, binFeatures, args);
    return base;
}

void PLTree::train(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args){
    rng.seed(args.seed);

    std::cerr << "Training tree ...\n";

    // Create tree structure
    if(!args.tree.empty()) loadTreeStructure(args.tree);
    else if(args.treeType == completeInOrder)
        buildCompleteTree(labels.cols(), args.arity, false);
    else if(args.treeType == completeRandom)
        buildCompleteTree(labels.cols(), args.arity, true);
    else buildKMeansTree(labels, features, args);

    // For stats
    int nCount = 0, yCount = 0;

    int rows = features.rows();
    assert(rows == labels.rows());
    assert(k >= labels.cols());

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
                nPositive.insert(n);
                while (n->parent) {
                    n = n->parent;
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

    std::ofstream out(args.model + "/weights.bin");
    if(args.threads > 1){
        // Run learning in parallel
        ThreadPool tPool(args.threads);
        std::vector<std::future<Base*>> results;

        for(const auto& n : tree)
            results.emplace_back(tPool.enqueue(nodeTrainThread, n->index, features.cols(),
                std::ref(binLabels[n->index]), std::ref(binFeatures[n->index]), std::ref(args)));

        // Saving in main thread
        for(int i = 0; i < results.size(); ++i) {
            printProgress(i, results.size());
            Base* base = results[i].get();
            base->save(out);
            delete base;
        }
    } else {
        for(int i = 0; i < tree.size(); ++i){
            printProgress(i, tree.size());
            Base base;
            base.train(features.cols(), binLabels[tree[i]->index], binFeatures[tree[i]->index], args);
            base.save(out);
        }
    }
    out.close();

    std::cerr << "  Points count: " << rows
                << "\n  Nodes per point: " << static_cast<float>(nCount) / rows
                << "\n  Labels per point: " << static_cast<float>(yCount) / rows
                << "\n";

    // Save data
    save(args.model + "/tree.bin");
    args.save(args.model + "/args.bin");
    std::cerr << "All done\n";
}

void PLTree::predict(std::vector<TreeNodeValue>& prediction, Feature* features, std::vector<Base*>& bases, int k){
    std::priority_queue<TreeNodeValue> nQueue;

    double val = bases[treeRoot->index]->predictProbability(features);
    //double val = -bases[treeRoot->index]->predictLoss(features);
    nQueue.push({treeRoot, val});

    while (!nQueue.empty()) {
        TreeNodeValue nVal = nQueue.top(); // Current node
        nQueue.pop();

        if(nVal.node->label >= 0){
            prediction.push_back({nVal.node, nVal.val}); // When using probability
            //prediction.push_back({nVal.node, exp(nVal.val)}); // When using loss
            if (prediction.size() >= k)
                break;
        } else {
            for(const auto& child : nVal.node->children){
                val = nVal.val * bases[child->index]->predictProbability(features); // When using probability
                //val = nVal.val - bases[child->index]->predictLoss(features); // When using loss
                nQueue.push({child, val});
            }
        }
    }
}

std::mutex testMutex;
int pointTestThread(PLTree* tree, Label* labels, Feature* features, std::vector<Base*>& bases,
    int k, std::vector<int>& correctAt){

    std::vector<TreeNodeValue> prediction;
    tree->predict(prediction, features, bases, k);

    testMutex.lock();
    for (int i = 0; i < k; ++i){
        int l = -1;
        while(labels[++l] > -1)
            if (prediction[i].node->label == labels[l]){
                ++correctAt[i];
                break;
            }
    }
    testMutex.unlock();

    return 0;
}

int batchTestThread(PLTree* tree, SRMatrix<Label>& labels, SRMatrix<Feature>& features,
    std::vector<Base*>& bases, int topK, int startRow, int stopRow, std::vector<int>& correctAt){

    std::vector<int> localCorrectAt (topK);
    for(int r = startRow; r < stopRow; ++r){
        std::vector<TreeNodeValue> prediction;
        tree->predict(prediction, features.row(r), bases, topK);

        for (int i = 0; i < topK; ++i)
            for (int j = 0; j < labels.size(r); ++j)
                if (prediction[i].node->label == labels.row(r)[j]){
                    ++localCorrectAt[i];
                    break;
                }
    }

    testMutex.lock();
    for (int i = 0; i < topK; ++i)
        correctAt[i] += localCorrectAt[i];
    testMutex.unlock();

    return 0;
}

void PLTree::test(SRMatrix<Label>& labels, SRMatrix<Feature>& features, std::vector<Base*>& bases, Args& args) {
    std::cerr << "Starting testing ...\n";

    std::vector<int> correctAt(args.topK);
    int rows = features.rows();
    assert(rows == labels.rows());

    if(args.threads > 1){
        // Run prediction in parallel

        // Pool
        ThreadPool tPool(args.threads);
        std::vector<std::future<int>> results;

        for(int r = 0; r < rows; ++r)
            results.emplace_back(tPool.enqueue(pointTestThread, this, labels.data()[r],
                features.data()[r], std::ref(bases), args.topK, std::ref(correctAt)));

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

            predict(prediction, features.data()[r], bases, args.topK);

            for (int i = 0; i < args.topK; ++i)
                for (int j = 0; j < labels.sizes()[r]; ++j)
                    if (prediction[i].node->label == labels.data()[r][j]){
                        ++correctAt[i];
                        break;
                    }
            printProgress(r, rows);
        }
    }

    double precisionAt = 0;
    for (int i = 0; i < args.topK; ++i) {
        precisionAt += correctAt[i];
        std::cerr << "P@" << i + 1 << ": " << precisionAt / (rows * (i + 1)) << "\n";
    }
    std::cerr << "All done\n";
}

void PLTree::loadTreeStructure(std::string file){
    std::cerr << "Loading PLTree structure from: " << file << "...\n";

    std::ifstream in(file);
    in >> k >> t;

    treeRoot = createTreeNode();
    for (int i = 1; i < t; ++i) createTreeNode();

    for (auto i = 0; i < t - 1; ++i) {
        int parent, child, label;
        in >> parent >> child >> label;

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

    assert(tree.size() == t);
    assert(treeLeaves.size() == k);
    std::cout << "  Nodes: " << tree.size() << ", leaves: " << treeLeaves.size() << "\n";
}

// Heuristic? Balanced K-Means clustering
void PLTree::balancedKMeans(std::vector<LabelsAssignation>* partition, SRMatrix<Feature>& labelsFeatures, Args &args){
    int labels = partition->size();
    int features = labelsFeatures.cols();
    int centroids = args.arity;

    //std::cerr << "balancedKMeans ...\n  Partition: " << partition->size() << ", centroids: " << centroids << "\n";

    int maxPartitionSize = static_cast<int>(ceil(static_cast<double>(labels) / centroids));

    // Test split - balanced tree
    /*
    for(int i = 0; i < partition->size(); ++i)
        (*partition)[i].value = i / maxPartitionSize;
    */

    // Init centroids
    std::vector<std::vector<double>> centroidsFeatures(centroids);
    for(int i = 0; i < centroids; ++i) {
        centroidsFeatures[i].resize(features, 0);
        std::uniform_int_distribution<int> dist(0, labels);
        setVector(labelsFeatures.row(dist(rng)), centroidsFeatures[i]);
    }

    double oldCos = INT_MIN, newCos = -1;

    std::vector<LabelsDistances> distances(labels);
    for(int i=0; i < labels; ++i ) distances[i].values.resize(centroids);

    while(newCos - oldCos >= args.kMeansEps){
        std::vector<int> centroidsSizes(centroids, 0);
        //std::cerr << "  newCos: " << newCos << ", oldCos: " << oldCos << "\n";

        // Calculate distances to centroids
        for(int i = 0; i < labels; ++i) {
            distances[i].index = i;
            double maxDist = INT_MIN;
            for(int j = 0; j < centroids; ++j) {
                distances[i].values[j].index = j;
                distances[i].values[j].value = labelsFeatures.dotRow((*partition)[i].index, centroidsFeatures[j]);
                if(distances[i].values[j].value > maxDist) maxDist = distances[i].values[j].value;
            }

            for(int j = 0; j < centroids; ++j)
                distances[i].values[j].value -= maxDist;

            std::sort(distances[i].values.begin(), distances[i].values.end());
        }

        // Assign labels to centroids and calculate new loss
        oldCos = newCos;
        newCos = 0;

        std::sort(distances.begin(), distances.end());

        for(int i = 0; i < labels; ++i){
            for(int j = 0; j < centroids; ++j){
                int cIndex = distances[i].values[j].index;
                int lIndex = distances[i].index;

                if(centroidsSizes[cIndex] < maxPartitionSize) {
                    (*partition)[lIndex].value = cIndex;
                    ++centroidsSizes[cIndex];
                    newCos += distances[i].values[j].value;
                    break;
                }
            }
        }

        newCos /= labels;

        // Update centroids
        for(int i = 0; i < centroids; ++i)
            std::fill(centroidsFeatures[i].begin(), centroidsFeatures[i].end(), 0);

        for(int i = 0; i < labels; ++i){
            int lIndex = (*partition)[i].index;
            int lCentroid = (*partition)[i].value;
            addVector(labelsFeatures.row(lIndex), centroidsFeatures[lCentroid]);
        }

        // Norm new centroids
        for(int i = 0; i < centroids; ++i)
            unitNorm(centroidsFeatures[i]);
    }
}

// TODO: Make it parallel
void PLTree::buildKMeansTree(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args &args){
    SRMatrix<Feature> labelsFeatures;

    std::cerr << "Building labels' features matrix ...\n";

    // Build labels' averaged features matrix
    {
        std::vector<std::unordered_map<int, double>> tmpLabelsFeatures(labels.cols());

        int rows = features.rows();
        assert(rows == labels.rows());

        for(int r = 0; r < rows; ++r){
            printProgress(r, rows);
            int rFeaturesSize = features.sizes()[r];
            int rLabelsSize = labels.sizes()[r];
            auto rFeatures = features.data()[r];
            auto rLabels = labels.data()[r];

            for (int i = 0; i < rFeaturesSize; ++i){
                for (int j = 0; j < rLabelsSize; ++j){
                    if (!tmpLabelsFeatures[rLabels[j]].count(rFeatures[i].index))
                        tmpLabelsFeatures[rLabels[j]][rFeatures[i].index] = 0;
                    tmpLabelsFeatures[rLabels[j]][rFeatures[i].index] += rFeatures[i].value;
                }
            }
        }

        for(int l = 0; l < labels.cols(); ++l){
            std::vector<Feature> labelFeatures;
            for(const auto& f : tmpLabelsFeatures[l])
                labelFeatures.push_back({f.first, f.second});
            std::sort(labelFeatures.begin(), labelFeatures.end());
            unitNorm(labelFeatures);
            labelsFeatures.appendRow(labelFeatures);
        }
    }

    std::cerr << "Hierarchical clustering ...\n";

    // Hierarchical K-means
    treeRoot = createTreeNode();
    k = labels.cols();

    std::vector<LabelsAssignation>* partition = new std::vector<LabelsAssignation>(k);
    for(int i = 0; i < k; ++i) (*partition)[i].index = i;
    std::queue<TreeNodePartition> nQueue;
    nQueue.push({treeRoot, partition});

    while(!nQueue.empty()) {
        TreeNodePartition nPart = nQueue.front(); // Current node
        nQueue.pop();

        if(nPart.partition->size() > args.maxLeaves){
            balancedKMeans(nPart.partition, labelsFeatures, args);
            std::vector<LabelsAssignation>** partitions = new std::vector<LabelsAssignation>*[args.arity];
            for(int i = 0; i < args.arity; ++i) partitions[i] = new std::vector<LabelsAssignation>();
            for(auto p : *nPart.partition) partitions[p.value]->push_back({p.index, 0});

            // Create children
            for(int i = 0; i < args.arity; ++i){
                TreeNode* n = createTreeNode(nPart.node);
                nQueue.push({n, partitions[i]});
            }
        } else
            for(int i = 0; i < nPart.partition->size(); ++i)
                createTreeNode(nPart.node, (*nPart.partition)[i].index);

        delete nPart.partition;
    }

    t = tree.size();
    assert(k == treeLeaves.size());
    std::cerr << "  Nodes: " << tree.size() << ", leaves: " << treeLeaves.size() << "\n";
}

void PLTree::buildCompleteTree(int labelCount, int arity, bool randomizeTree) {
    std::cerr << "Building complete PLTree ...\n";

    k = labelCount;
    t = static_cast<int>(ceil(static_cast<double>(arity * k - 1) / (arity - 1)));

    int ti = t - k;

    std::vector<int> labelsOrder;
    if (randomizeTree){
        for (auto i = 0; i < k; ++i) labelsOrder.push_back(i);
        std::shuffle(labelsOrder.begin(), labelsOrder.end(), rng);
    }

    for(size_t i = 0; i < t; ++i){
        int label = -1;
        TreeNode *parent = nullptr;

        if(i >= ti){
            if(randomizeTree) label = labelsOrder[i - ti];
            else label = i - ti;
        }

        if(i > 0) parent = tree[static_cast<int>(floor(static_cast<double>(i - 1) / arity))];
        createTreeNode(parent, label);
    }

    treeRoot = tree[0];
    std::cerr << "  Nodes: " << tree.size() << ", leaves: " << treeLeaves.size() << ", arity: " << arity << "\n";
}

TreeNode* PLTree::createTreeNode(TreeNode* parent, int label){
    TreeNode* n = new TreeNode();
    n->index = tree.size();
    n->label = label;
    n->parent = parent;
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
