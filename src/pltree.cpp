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
    // base->save(args.model + "/node_" + std::to_string(i) + ".bin", args);

    return base;
}

std::vector<std::vector<int>> splitLabels(std::vector<int> labels, const Args &args){
    std::vector<std::vector<int>> labelSplits;
    int partSize = ceil(float(labels.size()) / args.arity);
    std::vector<int>::const_iterator partBegin = labels.cbegin();
    while(partBegin < labels.cend()){
        std::vector<int> split =  std::vector<int>(partBegin, min(partBegin + partSize, labels.cend()));
        std::sort(split.begin(), split.end());;
        labelSplits.push_back(split);
        partBegin += partSize;
    }
    return labelSplits;
}

//std::vector<struct JobResult> processJob(int index, const std::vector<int> &jobInstances,
//                                         const std::vector<int> &jobLabels, std::ofstream &out,
//                                         SRMatrix<Label> &labels, SRMatrix<Feature> &features,
//                                         Args &args){
std::vector<struct JobResult> processJob(int index, const std::vector<int>& jobInstances,
                                         const std::vector<int>& jobLabels, std::ofstream& out,
                                         SRMatrix<Label>& labels, SRMatrix<Feature>& features,
                                         Args& args){
    //TODO add to params if trained multiple times
    int maxIter = 1000;
    int iter = 0;
    bool converged = false;
    std::vector<struct JobResult> results;

    std::vector<std::vector<int>> childPositiveInstances;
    std::vector<std::vector<int>> childLabels;
    std::vector<std::vector<double>> binLabelsChild;
    std::vector<Feature*> binFeatures;
    std::vector<Base*> childBases(args.arity);

//    for(std::vector<int>::const_iterator i = jobInstances.cbegin(); i != jobInstances.cend(); i++ ){
    for(int i = 0; i < jobInstances.size(); i++ ){
        binFeatures.push_back(features.data()[jobInstances[i]]);
    }

    int nodeArity;

    while(true) {
        //split labels
        childLabels = splitLabels(jobLabels, args);
        nodeArity = int(childLabels.size());
        assert(args.arity >= nodeArity);

        for(int i = 0; i <  nodeArity; i++){
            std::vector<double> cBinLabels;
            std::vector<int> cInstaces;

            for(std::vector<int>::const_iterator n = jobInstances.cbegin(); n != jobInstances.cend(); n++ ) {
                double binLabel = 0.0;
                for (int j = 0; j < labels.sizes()[*n]; ++j) {
                    auto label = labels.data()[*n][j];
                    //TODO use unordered_set instead
                    if (!binLabel and std::binary_search(childLabels[i].begin(), childLabels[i].end(), label)) {
                        binLabel = 1.0;
                        cInstaces.push_back(*n);
                        break;
                    }
                }
                cBinLabels.push_back(binLabel);
            }
            binLabelsChild.push_back(cBinLabels);
            childPositiveInstances.push_back(cInstaces);
        }

        //TODO remove old models if trained multiple times
        for(int i = 0; i <  nodeArity; i++){
            Base *base = new Base();
            base->train(features.cols(), binLabelsChild[i], binFeatures, args);
            childBases[i] = base;
        }

        // evaluate if convergence criteria are meet
        converged = true;

        if(converged or (iter++ > maxIter)) {
            break;
        }
    }

    for(int i = 0; i <  nodeArity; i++) {
        struct JobResult result{
                .base = childBases[i],
                .parent = index,
                .instances = childPositiveInstances[i],
                .labels = childLabels[i]
        };
        results.push_back(result);
    }
    return results;
}

struct JobResult PLTree::trainRoot(SRMatrix<Label> &labels, SRMatrix<Feature> &features, Args &args){
    std::vector<double> binLabels;
    std::vector<Feature*> binFeatures;
    std::vector<int> rootPositiveIndices;

    for(int r = 0; r < labels.rows(); ++r){
        binFeatures.push_back(features.data()[r]);
        if (labels.sizes()[r] > 0) {
            binLabels.push_back(1.0);
            rootPositiveIndices.push_back(r);
        } else {
            binLabels.push_back(0.0);
        }
    }

    Base *base = new Base();
    base->train(features.cols(), binLabels, binFeatures, args);

    JobResult result;
    result.parent = -1;
    result.instances = rootPositiveIndices;
    result.base = base;
    return result;
}

void PLTree::addModelToTree(Base *model, int parent, std::vector<int> &labels, std::vector<int> &instances,
                            std::ofstream &out, Args &args, std::vector<NodeJob> &nextLevelJobs){
    TreeNode *node = new TreeNode();
    tree.push_back(node);
    node->index = tree.size() - 1;
    model->save(out, args);
    delete model;


    if(parent == -1){ //ROOT
        node->parent = nullptr;
        treeRoot = tree[0];
        if(labels.size() > 0){
            node->label = -1;
            if(labels.size() > 1){
                nextLevelJobs.push_back(NodeJob{.parent = 0, .labels = labels, .instances = instances});
            }
        }
    } else {
        tree[parent]->children.push_back(node);
        node->parent = tree[parent];
        assert(tree[parent]->label == -1);

        if(labels.size() > 1) {
            nextLevelJobs.push_back(NodeJob{.parent = node->index, .labels = labels, .instances = instances });
            node->label = -1;
        } else {
            node->label = labels[0];
            treeLeaves[node->label] = node;
        }
    }
}


void PLTree::trainTopDown(SRMatrix<Label> &labels, SRMatrix<Feature> &features, Args &args){

    std::vector<struct NodeJob> jobs;
    std::vector<struct NodeJob> nextLevelJobs;

    std::ofstream out(args.model + "/weights.bin");

    struct JobResult rootResult = trainRoot(labels, features, args);
    std::vector<int> allLabels(labels.cols());//TODO: determine the number/list of unique labels/ in some correct way
    std::iota(allLabels.begin(), allLabels.end(), 0); // TODO: labels start from 0?
    addModelToTree(rootResult.base, -1, allLabels, rootResult.instances, out, args, jobs);
    //TODO remove copying of vectors
    if(args.threads > 1){
        // Run learning in parallel
        ThreadPool tPool(args.threads);
        while(jobs.size() != 0) {
            std::vector<std::future<std::vector<JobResult>>> levelResults;
            for (auto &job : jobs) {
                levelResults.emplace_back(
                        tPool.enqueue(processJob, job.parent, std::cref(job.instances), std::cref(job.labels),
                                      std::ref(out), std::ref(labels), std::ref(features), std::ref(args)));
            }

            for (int i = 0; i < levelResults.size(); ++i) {
                std::vector<JobResult> results;
                results = levelResults[i].get();

                for (auto result : results) {
                    addModelToTree(result.base, result.parent, result.labels, result.instances, out, args,
                                   nextLevelJobs);
                }
            }

            jobs = nextLevelJobs;
            nextLevelJobs.clear();
        }

    } else {
        while(jobs.size() != 0){
            nextLevelJobs.clear();
            for(auto job : jobs){
                std::vector<JobResult> results = processJob(job.parent, job.instances, job.labels, out, labels, features, args);
                for(auto result : results){
                    addModelToTree(result.base, result.parent, result.labels, result.instances,out, args, nextLevelJobs);
                }
                printProgress(job.parent, labels.cols());
            }
            jobs = nextLevelJobs;
            nextLevelJobs.clear();
        }
    }
    out.close();

    std::cerr<<std::endl<<"Training finished."<<std::endl;

    t = tree.size();
    k = treeLeaves.size();

    assert(k >= labels.cols());

    save(args.model + "/tree.bin");
    args.save(args.model + "/args.bin");
}


void PLTree::printTree(TreeNode *root){
//DFS
//    if(n->parent != nullptr){
//        std::cout<<"(index-"<<n->index<<":label-"<<n->label<<":parent-"<<n->parent->index<<")";
//    } else {
//        std::cout<<"(index-"<<n->index<<":label-"<<n->label<<":parent-"<<")";
//    }
//    for(auto c : n->children) printTree(c);

//BFS
    std::queue<int> q;
    q.push(0);
    int n;

    while(!q.empty()){
        n = q.front();
        q.pop();
        if(tree[n]->parent != nullptr){
            std::cout<<"(index-"<<tree[n]->index<<":label-"<<tree[n]->label<<":parent-"<<tree[n]->parent->index<<")";
        } else {
            std::cout<<"(index-"<<tree[n]->index<<":label-"<<tree[n]->label<<":parent-"<<")";
        }
        for(auto c : tree[n]->children){
            q.push(c->index);
        }
    }

}

void PLTree::train(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args){
    if(args.treeType == topDown){
        trainTopDown(labels, features, args);
    } else {
        trainFixed(labels, features, args);
    }
//    printTree(tree[0]);
//    std::cout<<std::endl;
}

void PLTree::trainFixed(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args){
    std::cerr << "Training tree ...\n";

    // Create tree structure
    if(args.tree.size() > 0) loadTreeStructure(args.tree);
    else if(args.treeType == completeInOrder)
        buildCompleteTree(labels.cols(), args.arity, false);
    else if(args.treeType == completeRandom)
        buildCompleteTree(labels.cols(), args.arity, true);
    else if(args.treeType == balancedInOrder)
        buildBalancedTree(labels.cols(), args.arity, false);
    else if(args.treeType == balancedRandom)
        buildBalancedTree(labels.cols(), args.arity, true);
    else buildTree(labels, features, args);

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

        int rSize = labels.sizes()[r];
        auto rLabels = labels.data()[r];

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

                for(auto child : n->children) {
                    if (nPositive.count(child)) nQueue.push(child);
                    else nNegative.insert(child);
                }
            }
        } else nNegative.insert(treeRoot);

        for (auto &n : nPositive){
            binLabels[n->index].push_back(1.0);
            binFeatures[n->index].push_back(features.data()[r]);
        }

        for (auto &n : nNegative){
            binLabels[n->index].push_back(0.0);
            binFeatures[n->index].push_back(features.data()[r]);
        }

        nCount += nPositive.size() + nNegative.size();
        yCount += labels.sizes()[r];
    }

    std::cerr << "Starting training in " << args.threads << " threads ...\n";

    std::ofstream out(args.model + "/weights.bin");
    if(args.threads > 1){
        // Run learning in parallel
        ThreadPool tPool(args.threads);
        std::vector<std::future<Base*>> results;

        for(auto n : tree)
            results.emplace_back(tPool.enqueue(nodeTrainThread, n->index, features.cols(),
                std::ref(binLabels[n->index]), std::ref(binFeatures[n->index]), std::ref(args)));

        for(int i = 0; i < results.size(); ++i) {
            Base* base = results[i].get();
            base->save(out, args);
            delete base;
            printProgress(i, results.size());
        }
    } else {
        for(int i = 0; i < tree.size(); ++i){
            Base base;
            base.train(features.cols(), binLabels[tree[i]->index], binFeatures[tree[i]->index], args);
            base.save(out, args);
            printProgress(i, tree.size());
        }
    }
    out.close();

    std::cerr << "  Points count: " << rows
                << "\n  Nodes per point: " << static_cast<float>(nCount) / rows
                << "\n  Labels per point: " << static_cast<float>(yCount) / rows;
    std::cerr << std::endl;

    // Save data
    save(args.model + "/tree.bin");
    args.save(args.model + "/args.bin");
}

void PLTree::predict(std::vector<TreeNodeValue>& prediction, Feature* features, std::vector<Base*>& bases, int k){
    std::priority_queue<TreeNodeValue> nQueue;

    //double val = bases[treeRoot->index]->predictProbability(features);
    double val = -bases[treeRoot->index]->predictLoss(features);
    nQueue.push({treeRoot, val});

    while (!nQueue.empty()) {
        TreeNodeValue nVal = nQueue.top(); // Current node
        nQueue.pop();

        if(nVal.node->label >= 0){
            //prediction.push_back({nVal.node, nVal.val}); // When using probability
            prediction.push_back({nVal.node, exp(nVal.val)}); // When using loss
            if (prediction.size() >= k)
                break;
        } else {
            for(auto child : nVal.node->children){
                //val = nVal.val * bases[child->index]->predictProbability(features); // When using probability
                val = nVal.val - bases[child->index]->predictLoss(features); // When using loss
                nQueue.push({child, val});
            }
        }
    }
}

std::mutex testMutex;
int pointTestThread(PLTree* tree, Label* labels, Feature* features, std::vector<Base*>& bases,
    int k, std::vector<int>& precision){

    std::vector<TreeNodeValue> prediction;
    tree->predict(prediction, features, bases, k);

    testMutex.lock();
    for (int i = 0; i < k; ++i){
        int l = -1;
        while(labels[++l] > -1)
            if (prediction[i].node->label == labels[l]) ++precision[i];
    }
    testMutex.unlock();

    return 0;
}

int batchTestThread(PLTree* tree, SRMatrix<Label>& labels, SRMatrix<Feature>& features,
    std::vector<Base*>& bases, int topK, int startRow, int stopRow, std::vector<int>& precision){

    std::vector<int> localPrecision (topK);
    for(int r = startRow; r < stopRow; ++r){
        std::vector<TreeNodeValue> prediction;
        tree->predict(prediction, features.data()[r], bases, topK);

        for (int i = 0; i < topK; ++i){
            for (int j = 0; j < labels.sizes()[r]; ++j) {
                if (prediction[i].node->label == labels.data()[r][j])
                    ++localPrecision[i];
            }
        }
    }

    testMutex.lock();
    for (int i = 0; i < topK; ++i)
        precision[i] += localPrecision[i];
    testMutex.unlock();

    return 0;
}

void PLTree::test(SRMatrix<Label>& labels, SRMatrix<Feature>& features, std::vector<Base*>& bases, Args& args) {
    std::cerr << "Starting testing ...\n";

    std::vector<int> precision (args.topK);
    int rows = features.rows();
    assert(rows == labels.rows());

    if(args.threads > 1){
        // Run prediction in parallel

        // Pool
        ThreadPool tPool(args.threads);
        std::vector<std::future<int>> results;

        for(int r = 0; r < rows; ++r)
            results.emplace_back(tPool.enqueue(pointTestThread, this, labels.data()[r],
                features.data()[r], std::ref(bases), args.topK, std::ref(precision)));

        for(int i = 0; i < results.size(); ++i) {
            results[i].get();
            printProgress(i, results.size());
        }

        // Batches
        /*
        ThreadSet tSet;
        int tRows = ceil(static_cast<double>(rows)/args.threads);
        for(int t = 0; t < args.threads; ++t)
            tSet.add(batchTestThread, this, std::ref(labels), std::ref(features), std::ref(bases),
                args.topK, t * tRows, std::min((t + 1) * tRows, labels.rows()), std::ref(precision));
        tSet.joinAll();
        */

    } else {
        std::vector<TreeNodeValue> prediction;
        for(int r = 0; r < rows; ++r){
            prediction.clear();
            predict(prediction, features.data()[r], bases, args.topK);

            for (int i = 0; i < args.topK; ++i){
                for (int j = 0; j < labels.sizes()[r]; ++j)
                    if (prediction[i].node->label == labels.data()[r][j]) ++precision[i];
            }
            printProgress(r, rows);
        }
    }

    double correct = 0;
    for (int i = 0; i < args.topK; ++i) {
        correct += precision[i];
        std::cerr << "P@" << i + 1 << ": " << correct / (rows * (i + 1)) << "\n";
    }
}

void PLTree::loadTreeStructure(std::string file){
    std::cerr << "Loading PLTree structure from: " << file << "...\n";

    std::ifstream in(file);
    in >> k >> t;

    for (auto i = 0; i < t; ++i) {
        TreeNode *n = new TreeNode();
        n->index = i;
        n->parent = nullptr;
        tree.push_back(n);
    }
    treeRoot = tree[0];

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
            childN->label = label;
            treeLeaves.insert(std::make_pair(childN->label, childN));
        }
    }
    in.close();

    std::cout << "  Nodes: " << tree.size() << ", leaves: " << treeLeaves.size() << "\n";

    assert(tree.size() == t);
    assert(treeLeaves.size() == k);
}

// K-means clustering
void PLTree::buildTree(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args &args){

}

void PLTree::buildBalancedTree(int labelCount, int arity, bool randomizeTree) {
    std::cerr << "Building balanced PLTree ...\n";

    std::default_random_engine rng(time(0));

    std::vector<int> labelsOrder;
    for (auto i = 0; i < labelCount; ++i) labelsOrder.push_back(i);
    if (randomizeTree){
        std::random_shuffle(labelsOrder.begin(), labelsOrder.end());
    }

    std::queue<std::tuple<std::vector<int>::const_iterator, std::vector<int>::const_iterator, int>> begin_end_parent;
    begin_end_parent.push(std::make_tuple(labelsOrder.cbegin(), labelsOrder.cend(), -1));
    int c;

    while(!begin_end_parent.empty()){
        auto bep = begin_end_parent.front();
        begin_end_parent.pop();
        auto begin = std::get<0>(bep);
        auto endd = std::get<1>(bep);
        auto parent = std::get<2>(bep);

        if(begin + 1 == endd){
            TreeNode *n = new TreeNode();
            n->index = tree.size();
            tree.push_back(n);
            n->label = *begin;
            treeLeaves[n->label] = n;
            if(parent != -1){
                tree[parent]->children.push_back(n);
                n->parent = tree[parent];
            }

        } else {
            TreeNode *n = new TreeNode();
            n->index = tree.size();
            tree.push_back(n);
            n->label = -1;
            if(parent != -1){
                tree[parent]->children.push_back(n);
                n->parent = tree[parent];
            }

            int partSize = ceil(float(endd - begin)/arity);
            std::vector<int>::const_iterator partBegin = begin;
            c = 0;
            while (partBegin < endd){
                assert(c++ < arity);
                begin_end_parent.push(std::make_tuple(partBegin, min(partBegin + partSize, endd), n->index));
                partBegin += partSize;
            }
        }
    }

    treeRoot = tree[0];
    treeRoot->parent = nullptr;

    k = treeLeaves.size();
    t = tree.size();

    std::cerr << "  Nodes: " << tree.size() << ", leaves: " << treeLeaves.size() << ", arity: " << arity << "\n";
}

void PLTree::buildCompleteTree(int labelCount, int arity, bool randomizeTree) {
    std::cerr << "Building complete PLTree ...\n";

    std::default_random_engine rng(time(0));
    k = labelCount;

    if (arity > 2) {
        double a = pow(arity, floor(log(k) / log(arity)));
        double b = k - a;
        double c = ceil(b / (arity - 1.0));
        double d = (arity * a - 1.0) / (arity - 1.0);
        double e = k - (a - c);
        t = static_cast<int>(e + d);
    } else {
        arity = 2;
        t = 2 * k - 1;
    }

    int ti = t - k;

    std::vector<int> labelsOrder;
    if (randomizeTree){
        for (auto i = 0; i < k; ++i) labelsOrder.push_back(i);
        std::random_shuffle(labelsOrder.begin(), labelsOrder.end());
    }

    for(size_t i = 0; i < t; ++i){
        TreeNode *n = new TreeNode();
        n->index = i;
        n->label = -1;

        if(i >= ti){
            if(randomizeTree) n->label = labelsOrder[i - ti];
            else n->label = i - ti;
            treeLeaves.insert(std::make_pair(n->label, n));
        }

        if(i > 0){
            n->parent = tree[static_cast<int>(floor(static_cast<double>(n->index - 1) / arity))];
            n->parent->children.push_back(n);
        }
        tree.push_back(n);
    }

    treeRoot = tree[0];
    treeRoot->parent = nullptr;

    std::cerr << "  Nodes: " << tree.size() << ", leaves: " << treeLeaves.size() << ", arity: " << arity << "\n";
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
