/*
 Copyright (c) 2019-2020 by Marek Wydmuch

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

#include "online_plt.h"
#include <cfloat>


OnlinePLT::OnlinePLT() {
    onlineTree = true;
    type = oplt;
    name = "Online PLT";
}

OnlinePLT::~OnlinePLT() {
    for (auto b : auxBases) delete b;
}

void OnlinePLT::init(int labelCount, Args& args) {
    tree = new Tree();

    if (args.treeType == onlineKaryRandom || args.treeType == onlineKaryComplete
        || args.treeType == onlineRandom || args.treeType == onlineBestScore)
        onlineTree = true;
    else
        tree->buildTreeStructure(labelCount, args);

    if (!onlineTree) {
        bases.resize(tree->t);
        for (auto& b : bases) {
            b = new Base(args);
            b->setupOnlineTraining(args);
        }
    }
}

void OnlinePLT::update(const int row, Label* labels, size_t labelsSize, Feature* features, size_t featuresSize,
                       Args& args) {
    UnorderedSet<TreeNode *> nPositive;
    UnorderedSet<TreeNode *> nNegative;
    
    if (onlineTree) { // Check if example contains a new label
        std::vector<int> newLabels;
        
        if(args.threads == 1) {
            for (int i = 0; i < labelsSize; ++i)
                if (!tree->leaves.count(labels[i])) newLabels.push_back(labels[i]);

            if (!newLabels.empty()) // Expand tree in case of the new label
                expandTree(newLabels, features, args);
        } else {
            {
                std::shared_lock<std::shared_timed_mutex> lock(treeMtx);
                for (int i = 0; i < labelsSize; ++i)
                    if (!tree->leaves.count(labels[i])) newLabels.push_back(labels[i]);
            }

            if (!newLabels.empty()) { // Expand tree in case of the new label
                std::unique_lock<std::shared_timed_mutex> lock(treeMtx);
                expandTree(newLabels, features, args);
            }
        }
    }

    // Update positive, negative and aux base estimators
    if(onlineTree && args.threads > 1) {
        {
            std::shared_lock<std::shared_timed_mutex> lock(treeMtx);
            getNodesToUpdate(nPositive, nNegative, labels, labelsSize);
        }

        for (const auto &n : nPositive) bases[n->index]->update(1.0, features, args);
        for (const auto &n : nNegative) bases[n->index]->update(0.0, features, args);
        for (const auto &n : nPositive) {
            if (auxBases[n->index] != nullptr)
                auxBases[n->index]->update(0.0, features, args);
        }
    }
    else {
        getNodesToUpdate(nPositive, nNegative, labels, labelsSize);
        for (const auto &n : nPositive) bases[n->index]->unsafeUpdate(1.0, features, args);
        for (const auto &n : nNegative) bases[n->index]->unsafeUpdate(0.0, features, args);
        if (onlineTree)
            for (const auto &n : nPositive) {
                if (auxBases[n->index] != nullptr)
                    auxBases[n->index]->unsafeUpdate(0.0, features, args);
            }
    }
}

void OnlinePLT::save(Args& args, std::string output) {

    // Save base classifiers
    std::ofstream out(joinPath(output, "weights.bin"));
    int size = bases.size();
    out.write((char*)&size, sizeof(size));
    for (int i = 0; i < size; ++i) {
        bases[i]->finalizeOnlineTraining(args);
        bases[i]->save(out);
    }
    out.close();

    // Save aux classifiers
//    out = std::ofstream(joinPath(output, "aux_weights.bin"));
//    size = bases.size();
//    out.write((char*)&size, sizeof(size));
//    for (int i = 0; i < size; ++i) {
//        if(auxBases[i] != nullptr) {
//            auxBases[i]->finalizeOnlineTraining(args);
//            auxBases[i]->save(out);
//        }
//        else {
//            out.write((char*)0, sizeof(int));
//            out.write((char*)0, sizeof(int));
//        }
//    }
//    out.close();

    // Save tree
    tree->saveToFile(joinPath(output, "tree.bin"));

    // Save tree structure
    tree->saveTreeStructure(joinPath(output, "tree.txt"));
}

TreeNode* OnlinePLT::createTreeNode(TreeNode* parent, int label, Base* base, Base* auxBase){
    auto n = tree->createTreeNode(parent, label);
    bases.push_back(base);
    auxBases.push_back(auxBase);

    n->subtreeLeaves = 0;

    return n;
}

void OnlinePLT::expandTree(const std::vector<Label>& newLabels, Feature* features, Args& args){

    //LOG(CERR) << "  New labels in size of " << newLabels.size() << " ...\n";

    std::default_random_engine rng(args.getSeed());
    std::uniform_int_distribution<uint32_t> dist(0, args.arity - 1);

    if (tree->nodes.empty()) // Empty tree
        tree->root = createTreeNode(nullptr, -1, new Base(args), nullptr);  // Root node doesn't need aux classifier

    if (tree->root->children.size() < args.arity) {
        TreeNode* newGroup = createTreeNode(tree->root, -1, new Base(args), new Base(args)); // Group node needs aux classifier
        for(const auto nl : newLabels)
            createTreeNode(newGroup, nl, new Base(args), nullptr);
        newGroup->subtreeLeaves += newLabels.size();
        tree->root->subtreeLeaves += newLabels.size();
        return;
    }

    TreeNode* toExpand = tree->root;

    //LOG(CERR) << "  Looking for node to expand ...\n";

    int depth = 0;
    float alfa = args.onlineTreeAlpha;
    while (auxBases[toExpand->index] == nullptr) { // Stop when we reach expandable node
        ++depth;

        if (args.treeType == onlineRandom)
            toExpand = toExpand->children[dist(rng)];

        else if (args.treeType == onlineBestScore) { // Best score
            //LOG(CERR) << "    Current node: " << toExpand->index << "\n";
            double bestScore = -DBL_MAX;
            TreeNode *bestChild;

            for (auto &child : toExpand->children) {
                double prob = bases[child->index]->predictProbability(features);
                double score = (1.0 - alfa) * prob + alfa * std::log(
                        (static_cast<double>(toExpand->subtreeLeaves) / toExpand->children.size()) / child->subtreeLeaves);
                if (score > bestScore) {
                    bestScore = score;
                    bestChild = child;
                }
                //LOG(CERR) << "      Child " << child->index << " score: " << score << ", prob: " << prob << "\n";
            }
            toExpand = bestChild;
        }

        toExpand->parent->subtreeLeaves += newLabels.size();
    }

    // Add labels
    for(int li = 0; li < newLabels.size(); ++li){
        Label nl = newLabels[li];
        if (toExpand->children.size() < args.maxLeaves) { // If there is still place in OVR
            ++toExpand->subtreeLeaves;
            auto newLabelNode = createTreeNode(toExpand, nl, auxBases[toExpand->index]->copy());
            //LOG(CERR) << "    Added node " << newLabelNode->index << " with label " << nl << " as " << toExpand->index << " child\n";
        } else {
            // If not, expand node
            bool inserted = false;

            //LOG(CERR) << "    Looking for other free siblings...\n";

            for (auto &sibling : toExpand->parent->children) {
                if (sibling->children.size() < args.maxLeaves && auxBases[sibling->index] != nullptr) {
                    auto newLabelNode = createTreeNode(sibling, nl, auxBases[sibling->index]->copy());
                    ++sibling->subtreeLeaves;
                    inserted = true;

                    //LOG(CERR) << "    Added node " << newLabelNode->index << " with label " << nl << " as " << sibling->index << " child\n";

                    break;
                }
            }

            if(inserted) continue;

            //LOG(CERR) << "    Expanding " << toExpand->index << " node to bottom...\n";

            // Create the new node for children and move leaves to the new node
            TreeNode* newParentOfChildren = createTreeNode(nullptr, -1, auxBases[toExpand->index]->copyInverted(), auxBases[toExpand->index]->copy());
            for (auto& child : toExpand->children) tree->setParent(child, newParentOfChildren);
            toExpand->children.clear();
            tree->setParent(newParentOfChildren, toExpand);
            newParentOfChildren->subtreeLeaves = toExpand->subtreeLeaves;

            // Create new branch with new node
            auto newBranch = createTreeNode(toExpand, -1, auxBases[toExpand->index]->copy(), new Base(args));
            createTreeNode(newBranch, nl, auxBases[toExpand->index]->copy(), nullptr);

            // Remove aux classifier
            if (toExpand->children.size() >= args.arity){
                delete auxBases[toExpand->index];
                auxBases[toExpand->index] = nullptr;
            }

            toExpand->subtreeLeaves += newLabels.size() - li;
            toExpand = newBranch;
            ++toExpand->subtreeLeaves;
        }
    }
}
