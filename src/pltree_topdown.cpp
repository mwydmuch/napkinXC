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
#include "utils.h"
#include "threads.h"

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

/*
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
*/

/*
void PLTree::buildTreeTopDown(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args &args){
    int n = features.rows(); // number of instances
    std::vector<int> active(0), left(0), right(0);

    for(int i=0; i < n; i++ ) active.push_back(i);

}

void PLTree::cut(SRMatrix<Label>& labels, SRMatrix<Feature>& features, std::vector<int>& active, std::vector<int>& left, std::vector<int>& right, Args &args){

}
*/
