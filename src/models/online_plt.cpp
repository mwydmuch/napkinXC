/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#include "online_plt.h"
#include <cfloat>


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
    args.treeType == onlineKMeans || args.treeType == onlineBestScore)
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
        std::vector<int> newLabels;

        std::unique_lock<std::shared_timed_mutex> lock(treeMtx);

        {
            //std::shared_lock<std::shared_timed_mutex> lock(treeMtx);
            for (int i = 0; i < labelsSize; ++i)
                if (!tree->leaves.count(labels[i])) newLabels.push_back(labels[i]);
        }

        if(!newLabels.empty()){
            // Expand tree in case of the new label

            expandTree(newLabels, features, args);

//            for(const auto& l : newLabels)
//                expandTree(l, features, args);

            std::cerr << "  Gathering as part of extension\n";
                getNodesToUpdate(nPositive, nNegative, labels, labelsSize);

            std::cerr << "  Updating as part of extension\n";

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

            return;
        }
    }

    {
        std::cerr << "  Gathering \n";
        std::shared_lock<std::shared_timed_mutex> lock(treeMtx);
        getNodesToUpdate(nPositive, nNegative, labels, labelsSize);
    }

    std::cerr << "  Updating\n";

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

TreeNode* OnlinePLT::createTreeNode(TreeNode* parent, int label, Base* base, Base* tmpBase){
    auto n = tree->createTreeNode(parent, label);
    bases.push_back(base);
    tmpBases.push_back(tmpBase);

    return n;
}

void OnlinePLT::expandTree(const std::vector<Label>& newLabels, Feature* features, Args& args){

    std::cerr << "  New labels in size of " << newLabels.size() << " ...\n";

    std::default_random_engine rng(args.getSeed());
    std::uniform_int_distribution<uint32_t> dist(0, args.arity - 1);

    if (tree->nodes.empty()) // Empty tree
        tree->root = createTreeNode(nullptr, -1, new Base(args), nullptr);  // Root node doesn't need tmp classifier

    if (tree->root->children.size() < args.arity) {
        TreeNode* newGroup = createTreeNode(tree->root, -1, new Base(args), new Base(args)); // Group node needs tmp classifier
        for(const auto nl : newLabels)
            createTreeNode(newGroup, nl, new Base(args), nullptr);
        return;
    }

    TreeNode* toExpand = tree->root;

    std::cerr << "  Looking for node to expand ...\n";

    int depth = 0;
    while (tmpBases[toExpand->index] == nullptr) { // Stop when we reach expandable node
        ++depth;

        if (args.treeType == onlineRandom)
            toExpand = toExpand->children[dist(rng)];

        else if (args.treeType == onlineBestScore) { // Best score
            std::cerr << "    Current node: " << toExpand->index << "\n";
            double bestScore = -DBL_MAX;
            TreeNode *bestChild;
            for (auto &child : toExpand->children) {
                double score = bases[child->index]->predictValue(features);
                if (score > bestScore) {
                    bestScore = score;
                    bestChild = child;
                }
                std::cerr << "      Child " << child->index << ": " << score << "\n";
            }
            toExpand = bestChild;
            ++toExpand->subtreeLeaves;
        }
    }

    // Add labels
    for(const auto nl : newLabels){
        if (toExpand->children.size() < args.maxLeaves) { // If there is still place in OVR
            auto newLabelNode = createTreeNode(toExpand, nl, tmpBases[toExpand->index]->copy());
            std::cerr << "    Added node " << newLabelNode->index << " with label " << nl << " as " << toExpand->index << " child\n";
        } else {
            // If not, expand node
            bool inserted = false;
            std::cerr << "    Looking for other free siblings...\n";
            for (auto &sibling : toExpand->parent->children) {
                if (sibling->children.size() < args.maxLeaves && tmpBases[sibling->index] != nullptr) {
                    auto newLabelNode = createTreeNode(sibling, nl, tmpBases[sibling->index]->copy());
                    inserted = true;

                    std::cerr << "    Added node " << newLabelNode->index << " with label " << nl << " as " << sibling->index << " child\n";

                    break;
                }
            }

            if(inserted) continue;

            std::cerr << "    Expanding " << toExpand->index << " node to bottom...\n";

            // Create the new node for children and move leaves to the new node
            TreeNode* newParentOfChildren = createTreeNode(nullptr, -1, tmpBases[toExpand->index]->copyInverted(), tmpBases[toExpand->index]->copy());
            for (auto& child : toExpand->children) tree->setParent(child, newParentOfChildren);
            toExpand->children.clear();
            tree->setParent(newParentOfChildren, toExpand);

            // Create new branch with new node
            auto newBranch = createTreeNode(toExpand, -1, tmpBases[toExpand->index]->copy(), new Base(args));
            createTreeNode(newBranch, nl, tmpBases[toExpand->index]->copy(), nullptr);

            // Remove temporary classifier
            if (toExpand->children.size() >= args.arity) {
                delete tmpBases[toExpand->index];
                tmpBases[toExpand->index] = nullptr;
            }

            toExpand = newBranch;
        }
    }

    tree->printTree();
    int x;
    std::cin >> x;
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
        tmpBases.push_back(nullptr); // Label node doesn't need tmp classifier
        tree->nextSubtree = firstLabel;
        return;
    }

    // Else
    TreeNode* toExpand = tree->root;

    int depth = 0;
    while (toExpand->children.size() >= args.arity && depth < args.maxDepth && toExpand->children[0]->label == -1) {
        ++depth;

        if (args.treeType == onlineBalanced)
            toExpand = toExpand->children[toExpand->subtreeLeaves++ % toExpand->children.size()];

        else if (args.treeType == onlineRandom)
            toExpand = toExpand->children[dist(rng)];

        else if (args.treeType == onlineBestScore) { // Best score
            //std::cerr << "  toExpand: " << toExpand->index << "\n";
            double bestScore = -DBL_MAX;
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
            ++toExpand->subtreeLeaves;
        }

        else if (args.treeType == onlineKMeans) { // Online K-Means tree
            //std::cerr << "  toExpand: " << toExpand->index << "\n";

            double bestScore = -DBL_MAX;
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
            ++toExpand->subtreeLeaves;
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
        // Check if one of siblings OVR is available
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
            Base* tb = tmpBases[toExpand->index];
            delete tb;
            tmpBases[toExpand->index] = nullptr;
        }

        //tree->printTree();
    }
}

