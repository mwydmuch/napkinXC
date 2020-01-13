/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#include "online_plt.h"

OnlinePLT::OnlinePLT() {
    onlineTree = true;
    type = oplt;
    name = "Online PLT";
}

OnlinePLT::~OnlinePLT() {
    for (auto b : tmpBases) delete b;
}

void OnlinePLT::init(int labelCount, Args& args) {
    tree = new Tree();

    if (args.treeType == onlineBalanced || args.treeType == onlineComplete || args.treeType == onlineRandom ||
    args.treeType == onlineKMeans || args.treeType == onlineBestScore || args.treeType == onlineBottomUp)
        onlineTree = true;
    else
        tree->buildTreeStructure(labelCount, args);

    if (!onlineTree) {
        bases.resize(tree->t);
        for (auto& b : bases) {
            b = new Base();
            b->setupOnlineTraining(args);
        }
    }
}

void OnlinePLT::update(const int row, Label* labels, size_t labelsSize, Feature* features, size_t featuresSize,
                       Args& args) {
    std::unordered_set<TreeNode*> nPositive;
    std::unordered_set<TreeNode*> nNegative;

    if (onlineTree) { // Check if example contains a new label
        std::lock_guard<std::mutex> lock(treeMtx);

        for (int i = 0; i < labelsSize; ++i)
            if (!tree->leaves.count(labels[i])) { // Expand tree in case of the new label
                if (args.newOnline) expandTree(labels[i], features, args);
                else expandTopDown(labels[i], features, args);
            }

        getNodesToUpdate(nPositive, nNegative, labels, labelsSize);
    } else
        getNodesToUpdate(nPositive, nNegative, labels, labelsSize);

    // Update positive base estimators
    for (const auto& n : nPositive) bases[n->index]->update(1.0, features, args);

    // Update negative
    for (const auto& n : nNegative) bases[n->index]->update(0.0, features, args);

    // Update temporary nodes
    if (onlineTree)
        for (const auto& n : nPositive){
            if(tmpBases[n->index] != nullptr)
                tmpBases[n->index]->update(0.0, features, args);
        }

    // Update centroids
    if (args.treeType == onlineKMeans){
        //std::cerr << "  Updateing centroid\n";
        for (const auto& n : nPositive){
            if(n->label == -1 || n->index != 0){
                //std::cerr << "  For node " << n->index << "\n";
                // Update centroids
                UnorderedMap<int, float> &map = n->centroid;
                Feature *f = features;
                while (f->index != -1) {
                    if (f->index == 1){
                        ++f;
                        continue;
                    }
                    int index = f->index;
                    if (args.kMeansHash) index = hash(f->index) % args.hash;
                    map[index] += f->value;
                    ++f;
                }

                n->norm = 0;
                for(const auto& w : map)
                    n->norm += w.second * w.second;
                n->norm = std::sqrt(n->norm);
            }
        }
    }
}

void OnlinePLT::save(Args& args, std::string output) {

    // Save base classifiers
    std::ofstream out(joinPath(output, "weights.bin"));
    int size = bases.size();
    out.write((char*)&size, sizeof(size));
    for (int i = 0; i < bases.size(); ++i) {
        bases[i]->finalizeOnlineTraining(args);
        bases[i]->save(out);
    }
    out.close();

    // Save tree
    tree->saveToFile(joinPath(output, "tree.bin"));

    // Save tree structure
    tree->saveTreeStructure(joinPath(output, "tree"));
}

void OnlinePLT::expandTree(Label newLabel, Feature* features, Args& args){
    //std::cerr << "  New label " << newLabel << "\n";

    std::default_random_engine rng(args.getSeed());
    std::uniform_int_distribution<uint32_t> dist(0, args.arity - 1);

    if (tree->nodes.empty()) { // Empty tree
        tree->root = tree->createTreeNode(nullptr);
        bases.push_back(new Base());
        bases.back()->setupOnlineTraining(args);
        tmpBases.push_back(new Base());
        tmpBases.back()->setupOnlineTraining(args);
        TreeNode* firstLabel = tree->createTreeNode(tree->root, newLabel);
        bases.push_back(new Base());
        bases.back()->setupOnlineTraining(args);
        tmpBases.push_back(nullptr); // Label node doesn't need tmp classfier
        tree->nextSubtree = firstLabel;
        return;
    }

    // Else
    TreeNode* toExpand = tree->root;

    while (toExpand->children.size() >= args.arity && toExpand->children[0]->label == -1) {
        if (args.treeType == onlineBalanced)
            toExpand = toExpand->children[toExpand->nextToExpand++ % toExpand->children.size()];

        else if (args.treeType == onlineRandom)
            toExpand = toExpand->children[dist(rng)];

        else if (args.treeType == onlineBestScore) { // Best score
            //std::cerr << "  toExpand: " << toExpand->index << "\n";
            double bestScore = INT_MIN;
            TreeNode *bestChild;
            for (auto &child : toExpand->children) {
                double score = bases[child->index]->predictValue(features);
                if (score > bestScore) {
                    bestScore = score;
                    bestChild = child;
                }
                //std::cerr << "  child: " << child->index << " " << score << "\n";
            }
            //std::cerr << bestScore << " " << bestChild->index << "\n";
            toExpand = bestChild;
        }

        else if (args.treeType == onlineKMeans) { // Online K-Means tree
            //std::cerr << "  toExpand: " << toExpand->index << "\n";

            double bestScore = INT_MIN;
            TreeNode *bestChild;
            for (auto &child : toExpand->children) {
                double score = 0.0;
                UnorderedMap<int, float> &map = child->centroid;

                Feature *f = features;
                while (f->index != -1) {
                    if (f->index == 1){
                        ++f;
                        continue;
                    }
                    int index = f->index;
                    if(args.kMeansHash) index = hash(f->index) % args.hash;
                    auto w = map.find(index);
                    if (w != map.end()) score += (w->second / child->norm) * f->value;
                    ++f;
                }

                if (score > bestScore) {
                    bestScore = score;
                    bestChild = child;
                }

                //std::cerr << "  child: " << child->index << " " << score << "\n";
            }
            //std::cerr << bestScore << " " << bestChild->index << "\n";
            toExpand = bestChild;
        }
    }

    // Expand selected node
    //std::cerr << "  Node " << toExpand->index << " selected to expand\n";

    // If number of children have not reached max leaves
    if (toExpand->children.size() < args.maxLeaves) {
        TreeNode* newLabelNode = tree->createTreeNode(toExpand, newLabel);
        bases.push_back(tmpBases[toExpand->index]->copy());
        tmpBases.push_back(nullptr);
        //std::cerr << "  Added as " << newLabelNode->index << " as " << toExpand->index << " child\n";
        //tree->printTree();
    } else {
        // Check if one of siblings OVR is avaialble
        if(toExpand->parent) {
            //std::cerr << "  Looking for other free siblings...\n";
            for (auto &sibling : toExpand->parent->children) {
                if (sibling->children.size() < args.maxLeaves && sibling->children[0]->label != -1) {
                    toExpand = sibling;

                    TreeNode *newLabelNode = tree->createTreeNode(toExpand, newLabel);
                    bases.push_back(tmpBases[toExpand->index]->copy());
                    tmpBases.push_back(nullptr);

                    //std::cerr << "  Added as " << newLabelNode->index << " as " << toExpand->index << " child\n";
                    //tree->printTree();

                    return;
                }
            }
        }

        // If not, expand node
        //std::cerr << "  Expanding " << toExpand->index << " node to bottom...\n";

        TreeNode* newParentOfChildren = tree->createTreeNode();
        bases.push_back(tmpBases[toExpand->index]->copyInverted());
        tmpBases.push_back(tmpBases[toExpand->index]->copy());

        // Move leaves to the new node
        for (auto& child : toExpand->children) tree->setParent(child, newParentOfChildren);
        toExpand->children.clear();
        tree->setParent(newParentOfChildren, toExpand);

        // Create new branch
        TreeNode* newBranch = tree->createTreeNode(toExpand);
        bases.push_back(tmpBases[toExpand->index]->copy());
        tmpBases.push_back(new Base());
        tmpBases.back()->setupOnlineTraining(args);

        // Create new node
        TreeNode* newLabelNode = tree->createTreeNode(newBranch, newLabel);
        bases.push_back(tmpBases[toExpand->index]->copy());
        tmpBases.push_back(nullptr);

        // Remove temporary classifier
        if (toExpand->children.size() == args.arity - 1) {
            delete tmpBases[toExpand->index];
            tmpBases[toExpand->index] = nullptr;
        }

        //tree->printTree();
    }
}

void OnlinePLT::expandTopDown(Label newLabel, Feature* features, Args& args) {

    if (tree->nodes.empty()) { // Empty tree
        tree->root = tree->createTreeNode(nullptr, newLabel);
        bases.push_back(new Base());
        bases.back()->setupOnlineTraining(args);
        tmpBases.push_back(new Base());
        tmpBases.back()->setupOnlineTraining(args);
        tree->nextSubtree = tree->root;
        return;
    }

    TreeNode* toExpand = tree->root;
    int depth = 0;

    // Select node based on policy
    if (args.treeType == onlineBalanced) { // Balanced policy
        while (toExpand->children.size() >= args.arity) {
            toExpand = toExpand->children[toExpand->nextToExpand++ % toExpand->children.size()];
            ++depth;
        }
    } else if (args.treeType == onlineComplete) { // Complete
        if (tree->nodes[tree->nextToExpand]->children.size() >= args.arity) ++tree->nextToExpand;
        toExpand = tree->nodes[tree->nextToExpand];
    } else if (args.treeType == onlineRandom) { // Random policy
        std::default_random_engine rng(args.getSeed());
        std::uniform_int_distribution<uint32_t> dist(0, args.arity - 1);
        while (toExpand->children.size() >= args.arity){
            toExpand = toExpand->children[dist(rng)];
            ++depth;
        }
    } else if (args.treeType == onlineBestScore) { // Best score
        while (toExpand->children.size() >= args.arity && depth < args.maxDepth) {
            std::cerr << "  toExpand: " << toExpand->index << "\n";
            double bestScore = INT_MIN;
            TreeNode* bestChild;
            for(auto& child : toExpand->children){
                double score = bases[child->index]->predictValue(features);
                if(score > bestScore) {
                    bestScore = score;
                    bestChild = child;
                }
                std::cerr << "  child: " << child->index << " " << score << "\n";
            }
            //std::cerr << bestScore << " " << bestChild->index << "\n";
            toExpand = bestChild;
        }
    } else if (args.treeType == onlineKMeans) { // Online K-Means tree
        while (toExpand->children.size() >= args.arity && depth < args.maxDepth
        ) {
            double bestScore = INT_MIN;
            TreeNode *bestChild;
            for (auto &child : toExpand->children) {
                double score = 0.0;
                UnorderedMap<int, float> &map = child->centroid;

                Feature *f = features;
                while (f->index != -1) {
                    if (f->index == 1){
                        ++f;
                        continue;
                    }
                    int index = f->index;
                    if(args.kMeansHash) index = hash(f->index) % args.hash;
                    auto w = map.find(f->index);
                    if (w != map.end()) score += w->second / child->norm * f->value;
                    ++f;
                }

                if (score > bestScore) {
                    bestScore = score;
                    bestChild = child;
                }
            }
            toExpand = bestChild;
        }
    }

    // Expand selected node
    if (toExpand->children.size() == 0) {
        TreeNode* parentLabelNode = tree->createTreeNode(toExpand, toExpand->label);
        bases.push_back(tmpBases[toExpand->index]->copyInverted());
        tmpBases.push_back(tmpBases[toExpand->index]->copy());
    }

    TreeNode* newLabelNode = tree->createTreeNode(toExpand, newLabel);
    bases.push_back(tmpBases[toExpand->index]->copyInverted());
    tmpBases.push_back(new Base());
    tmpBases.back()->setupOnlineTraining(args);

    // Remove temporary classifier
    if (toExpand->children.size() == args.arity - 1) {
        delete tmpBases[toExpand->index];
        tmpBases[toExpand->index] = nullptr;
    }
}

void OnlinePLT::expandBottomUp(Label newLabel, Args& args) {
    TreeNode* nextSubtree = tree->nextSubtree;
    if (nextSubtree->children.size() < args.arity) {
        if (nextSubtree->label != -1) {
            TreeNode* labelChild = tree->createTreeNode(nextSubtree, nextSubtree->label);
            bases.push_back(tmpBases[nextSubtree->index]->copyInverted());
            tmpBases.push_back(new Base());
            tmpBases.back()->setupOnlineTraining(args);
            ++nextSubtree->subtreeDepth;
        }

        TreeNode* newChild = tree->createTreeNode(nextSubtree, newLabel);
        bases.push_back(tmpBases[nextSubtree->index]->copy());
        tmpBases.push_back(new Base());
        tmpBases.back()->setupOnlineTraining(args);

        if (nextSubtree->parent && nextSubtree->children.size() == args.arity - 1 &&
                nextSubtree->parent->subtreeDepth == nextSubtree->subtreeDepth + 1) {
            delete tmpBases[nextSubtree->index];
            tmpBases[nextSubtree->index] = nullptr;
        }

        if (nextSubtree->subtreeDepth > 2) tree->nextSubtree = newChild;
    } else if (nextSubtree->parent && nextSubtree->parent->subtreeDepth == nextSubtree->subtreeDepth + 1) {
        // Moving up
        tree->nextSubtree = nextSubtree->parent;
        expandBottomUp(newLabel, args);
    } else {
        // Expanding subtree
        TreeNode* parentOfOldTree = tree->createTreeNode();
        bases.push_back(tmpBases[nextSubtree->index]->copy());
        tmpBases.push_back(new Base());
        tmpBases.back()->setupOnlineTraining(args);

        parentOfOldTree->subtreeDepth = nextSubtree->subtreeDepth;
        tree->moveSubtree(nextSubtree, parentOfOldTree);
        ++nextSubtree->subtreeDepth;

        expandBottomUp(newLabel, args);
    }
}
