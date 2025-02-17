/*
 Copyright (c) 2018-2020 by Marek Wydmuch, Kalina Jasinska-Kobus, Robert Istvan Busa-Fekete

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
#include <chrono>
#include <climits>
#include <cmath>
#include <iomanip>
#include <list>
#include <random>
#include <sstream>
#include <vector>

#include "label_tree.h"
#include "threads.h"

LabelTree::LabelTree() {

}

LabelTree::~LabelTree() {
    clear();
}

void LabelTree::clear() {
    for (auto n : nodes) delete n;
    nodes.clear();
    leaves = UnorderedMap<int, TreeNode*>();
}

void LabelTree::buildTreeStructure(int labelCount, Args& args) {
    clear();

    // Create a tree structure
    Log(CERR) << "Building tree ...\n";

    if (args.treeType == completeKaryInOrder)
        buildCompleteTree(labelCount, false, args);
    else if (args.treeType == completeKaryRandom)
        buildCompleteTree(labelCount, true, args);
    else if (args.treeType == balancedInOrder)
        buildBalancedTree(labelCount, false, args);
    else if (args.treeType == balancedRandom)
        buildBalancedTree(labelCount, true, args);
    else if (args.treeType < custom)
        throw std::invalid_argument("This tree type is not supported for this model type");
    else if (args.treeType > custom)
        throw std::invalid_argument("Unknown tree type");
}

void LabelTree::buildTreeStructure(SRMatrix& labels, SRMatrix& features, Args& args) {
    clear();

    // Load tree structure from file
    if (!args.treeStructure.empty()) loadTreeStructure(args.treeStructure);

    // Create a tree structure
    Log(CERR) << "Building tree ...\n";

    if (args.treeType == completeKaryInOrder)
        buildCompleteTree(labels.cols(), false, args);
    else if (args.treeType == completeKaryRandom)
        buildCompleteTree(labels.cols(), true, args);
    else if (args.treeType == balancedInOrder)
        buildBalancedTree(labels.cols(), false, args);
    else if (args.treeType == balancedRandom)
        buildBalancedTree(labels.cols(), true, args);
    else if (args.treeType == huffman)
        buildHuffmanTree(labels, args);
    else if (args.treeType == hierarchicalKmeans) {
        SRMatrix labelsFeatures;
        computeLabelsFeaturesMatrix(labelsFeatures, labels, features, args.threads, args.norm,
                                    args.kmeansWeightedFeatures);
        //labelsFeatures.dump(joinPath(args.output, "lf_mat.txt"));
        buildKmeansTree(labelsFeatures, args);
    } else if (args.treeType == onlineKaryComplete || args.treeType == onlineKaryRandom)
        buildOnlineTree(labels, features, args);
    else if (args.treeType < custom)
        buildOnlineTree(labels, features, args);
    else if (args.treeType != custom)
        throw std::invalid_argument("Unknown tree type");

    if(args.flattenTree) flattenTree(args.flattenTree);

    //printTree();
    //validateTree();
    Log(CERR) << "  Nodes: " << nodes.size() << ", leaves: " << leaves.size() << "\n";
}

TreeNodePartition LabelTree::buildKmeansTreeThread(TreeNodePartition nPart, SRMatrix& labelsFeatures, Args& args,
                                              int seed) {
    kmeans(nPart.partition, labelsFeatures, args.arity, args.kmeansEps, args.kmeansBalanced, seed);
    return nPart;
}

void LabelTree::buildKmeansTree(SRMatrix& labelsFeatures, Args& args) {
    Log(CERR) << "Hierarchical K-Means clustering in " << args.threads << " threads ...\n";

    root = createTreeNode();
    int k = labelsFeatures.rows();

    long seed = args.getSeed();
    std::default_random_engine rng(seed);
    std::uniform_int_distribution<int> kmeansSeeder(0, INT_MAX);

    auto partition = new std::vector<Assignation>(k);
    for (int i = 0; i < k; ++i) (*partition)[i].index = i;

    // Run clustering in parallel
    ThreadPool tPool(args.threads);
    std::vector<std::future<TreeNodePartition>> results;

    TreeNodePartition rootPart = {root, partition};
    results.emplace_back(
        tPool.enqueue(buildKmeansTreeThread, rootPart, std::ref(labelsFeatures), std::ref(args), kmeansSeeder(rng)));

    for (int r = 0; r < results.size(); ++r) {
        // Enqueuing new clustering tasks in the main thread ensures determinism
        TreeNodePartition nPart = results[r].get();

        // This needs to be done this way in case of imbalanced K-Means
        auto partitions = new std::vector<Assignation>*[args.arity];
        for (int i = 0; i < args.arity; ++i) partitions[i] = new std::vector<Assignation>();
        for (auto a : *nPart.partition) partitions[a.value]->push_back({a.index, 0});

        // Create children
        for (int i = 0; i < args.arity; ++i) {
            if (partitions[i]->empty())
                continue;
            else if (partitions[i]->size() == 1) {
                createTreeNode(nPart.node, partitions[i]->front().index);
                delete partitions[i];
                continue;
            }

            TreeNode* n = createTreeNode(nPart.node);

            if (partitions[i]->size() <= args.maxLeaves) {
                for (const auto& a : *partitions[i]) createTreeNode(n, a.index);
                delete partitions[i];
            } else {
                TreeNodePartition childPart = {n, partitions[i]};
                results.emplace_back(tPool.enqueue(buildKmeansTreeThread, childPart, std::ref(labelsFeatures),
                                                   std::ref(args), kmeansSeeder(rng)));
            }
        }

        delete nPart.partition;
    }
}

void LabelTree::buildHuffmanTree(SRMatrix& labels, Args& args) {
    Log(CERR) << "Building Huffman Tree ...\n";

    int k = labels.cols();
    auto labelsProb = computeLabelsPriors(labels);

    std::priority_queue<TreeNodeValue, std::vector<TreeNodeValue>, std::greater<>> probQueue;
    for (int i = 0; i < k; i++) {
        TreeNode* n = createTreeNode(nullptr, i);
        probQueue.push({n, labelsProb[i].value});
    }

    while (!probQueue.empty()) {
        std::vector<TreeNodeValue> toMerge;
        for (int a = 0; a < args.arity; ++a) {
            toMerge.push_back(probQueue.top());
            probQueue.pop();
            if (probQueue.empty()) break;
        }

        TreeNode* parent = createTreeNode();
        Real aggregatedProb = 0;
        for (TreeNodeValue& e : toMerge) {
            e.node->parent = parent;
            parent->children.push_back(e.node);
            aggregatedProb += e.value;
        }

        if (probQueue.empty())
            root = parent;
        else
            probQueue.push({parent, aggregatedProb});
    }
}

void LabelTree::buildBalancedTree(int labelCount, bool randomizeOrder, Args& args) {
    Log(CERR) << "Building balanced Tree ...\n";

    root = createTreeNode();
    int k = labelCount;
    std::default_random_engine rng(args.seed);

    auto partition = new std::vector<Assignation>(k);
    for (int i = 0; i < k; ++i) (*partition)[i].index = i;

    if (randomizeOrder) std::shuffle(partition->begin(), partition->end(), rng);

    std::queue<TreeNodePartition> nQueue;
    nQueue.push({root, partition});

    while (!nQueue.empty()) {
        TreeNodePartition nPart = nQueue.front(); // Current node
        nQueue.pop();
        if (nPart.partition->size() > args.maxLeaves) {
            auto partitions = new std::vector<Assignation>*[args.arity];
            for (int i = 0; i < args.arity; ++i) partitions[i] = new std::vector<Assignation>();

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
                TreeNode* n = createTreeNode(nPart.node);
                nQueue.push({n, partitions[i]});
            }
        } else
            for (const auto& a : *nPart.partition) createTreeNode(nPart.node, a.index);

        delete nPart.partition;
    }
}

void LabelTree::buildCompleteTree(int labelCount, bool randomizeOrder, Args& args) {
    Log(CERR) << "Building complete Tree ...\n";

    std::default_random_engine rng(args.getSeed());

    int k = labelCount;
    int t = static_cast<int>(ceil(static_cast<Real>(args.arity * k - 1) / (args.arity - 1)));

    int ti = t - k;

    std::vector<int> labelsOrder;
    if (randomizeOrder) {
        labelsOrder.resize(k);
        for (auto i = 0; i < k; ++i) labelsOrder[i] = i;
        std::shuffle(labelsOrder.begin(), labelsOrder.end(), rng);
    }

    root = createTreeNode();
    for (size_t i = 1; i < t; ++i) {
        int label = -1;
        if (i >= ti) {
            if (randomizeOrder)
                label = labelsOrder[i - ti];
            else
                label = i - ti;
        }

        TreeNode* parent = nodes[static_cast<int>(floor(static_cast<Real>(i - 1) / args.arity))];
        createTreeNode(parent, label);
    }
}

void LabelTree::buildOnlineTree(SRMatrix& labels, SRMatrix& features, Args& args) {
    Log(CERR) << "Building online tree ...\n";

    int nextToExpand = 0;

    int rows = features.rows();
    for (int r = 0; r < rows; ++r) {
        printProgress(r, rows);

        auto rSize = labels.size(r);
        auto& rLabels = labels[r];

        // Check row
        for (int i = 0; i < rSize; ++i) {
            if (!leaves.count(rLabels[i])) {

                int newLabel = rLabels[i];

                if (nodes.empty()) { // Empty tree
                    root = createTreeNode(nullptr, newLabel);
                    continue;
                }

                TreeNode* toExpand = root;

                // Select node based on policy
                if (args.treeType == onlineKaryComplete) { // Complete policy
                    if (nodes[nextToExpand]->children.size() >= args.arity) ++nextToExpand;
                    toExpand = nodes[nextToExpand];
                } else if (args.treeType == onlineKaryRandom) { // Random policy
                    std::default_random_engine rng(args.getSeed());
                    std::uniform_int_distribution<uint32_t> dist(0, args.arity - 1);
                    while (toExpand->children.size() == args.arity) toExpand = toExpand->children[dist(rng)];
                } else
                    throw std::invalid_argument("Unknown tree type");

                // Expand selected node
                if (toExpand->children.empty()) TreeNode* parentLabelNode = createTreeNode(toExpand, toExpand->label);
                TreeNode* newLabelNode = createTreeNode(toExpand, newLabel);
            }
        }
    }
}

void LabelTree::loadTreeStructure(std::string file) {
    Log(CERR) << "Loading tree structure from: " << file << "...\n";

    std::vector<std::tuple<int, int, int>> treeStructure;
    std::ifstream in(file);
    int k, t;
    in >> k >> t;
    treeStructure.reserve(t);

    if (k >= t) throw std::invalid_argument("The specified number of labels = " + std::to_string(k) + " is higher than the specified number of nodes = " + std::to_string(t));

    std::string line;
    while (std::getline(in, line)) {
        if (!line.length()) continue;

        int parent, child, label = -1;
        std::string sLabel;

        std::istringstream lineISS(line);
        lineISS >> parent >> child >> sLabel;
        if (sLabel.size()) label = std::stoi(sLabel);
        treeStructure.emplace_back(parent, child, label);
    }
    in.close();
    setTreeStructure(treeStructure);
}

void LabelTree::setTreeStructure(std::vector<std::tuple<int, int, int>> treeStructure){
    clear();

    int t = treeStructure.size();
    root = nullptr;
    for (int i = 0; i < t; ++i) createTreeNode();

    int k = 0;
    for(auto &tn : treeStructure) {
        if(std::get<2>(tn) != -1) ++k;
    }

    nodes.reserve(t);
    leaves.reserve(k);

    for(auto &tn : treeStructure){
        int parent, child, label;
        std::tie(parent, child, label) = tn;

        if (child >= t) throw std::invalid_argument("The node index = " + std::to_string(child) + " is higher than the number of nodes = " + std::to_string(t));
        if (parent >= t) throw std::invalid_argument("The parent index = " + std::to_string(parent) + " is higher than the number of nodes = " + std::to_string(t));
        if (label >= k) throw std::invalid_argument("The label index = " + std::to_string(label) + " is higher than the number of labels = " + std::to_string(k));

        TreeNode* n = nodes[child];
        if (parent == -1){
            if(root != nullptr) throw std::invalid_argument("More than one root node");
            root = nodes[child];
        } else {
            TreeNode* parentN = nodes[parent];
            parentN->children.push_back(n);
            n->parent = parentN;
        }

        if (label >= 0) {
            assert(leaves.count(label) == 0);
            assert(label < k);
            n->label = label;
            leaves[n->label] = n;
        }
    }

    validateTree();

    assert(nodes.size() == t);
    assert(leaves.size() == k);
}

void LabelTree::saveTreeStructure(std::string file) {
    Log(CERR) << "Saving tree structure to: " << file << "...\n";

    std::ofstream out(file);
    out << leaves.size() << " " << nodes.size() << "\n";
    for (auto& n : nodes) {
        if (n->parent != nullptr) out << n->parent->index;
        else out << -1;
        out << " " << n->index << " ";
        if (n->label >= 0) out << n->label;
        // else out << -1;
        out << "\n";
    }
    out.close();
}

std::vector<std::tuple<int, int, int>> LabelTree::getTreeStructure() {
    std::vector<std::tuple<int, int, int>> treeStructure;
    treeStructure.reserve(nodes.size());
    for (auto& n : nodes) {
        int parent = -1;
        if (n->parent != nullptr) parent = n->parent->index;
        treeStructure.emplace_back(parent, n->index, n->label);
    }

    return treeStructure;
}

void LabelTree::validateTree() {
    for (const auto& n : nodes) {
        if (n->parent == nullptr && n != root)
            throw std::invalid_argument("A node without a parent that is not a tree root exists");
        if (n->children.size() == 0 && n->label < 0)
            throw std::invalid_argument("An internal node without children exists");
    }
}

void LabelTree::flattenTree(int levels){
    Log(CERR) << "Flattening tree structure ...\n";

    UnorderedSet<TreeNode*> currentLevel;
    UnorderedSet<TreeNode*> nextLevel;
    currentLevel.reserve(leaves.size());
    for(auto l : leaves) currentLevel.insert(l.second->parent);

    while(currentLevel.size() > 1){
        for(auto n : currentLevel) {
            auto nParent = n->parent;
            for (int i = 0; i < levels; ++i) {
                if(nParent->parent != nullptr) nParent = nParent->parent; // If root not reached
                else break;
            }

            if(!nextLevel.count(nParent)) nParent->children.clear();
            nParent->children.push_back(n);
            n->parent = nParent;
            nextLevel.insert(nParent);
        }

        currentLevel = nextLevel;
        nextLevel = UnorderedSet<TreeNode*>();
    }

    reenumerateNodes();
}

void LabelTree::reenumerateNodes(){
    std::vector<TreeNode*> newNodes;
    UnorderedSet<TreeNode*> toKeep;

    std::queue<TreeNode*> nQueue;
    nQueue.push(root);

    while (!nQueue.empty()) {
        TreeNode* n = nQueue.front();
        nQueue.pop();

        n->index = newNodes.size();
        newNodes.push_back(n);
        toKeep.insert(n);

        for (auto c : n->children) nQueue.push(c);
    }

    for(auto n : nodes) if(!toKeep.count(n)) delete n;
    nodes = newNodes;
}

TreeNode* LabelTree::createTreeNode(TreeNode* parent, int label) {
    TreeNode* n = new TreeNode();
    n->index = nodes.size();
    nodes.push_back(n);
    setLabel(n, label);
    setParent(n, parent);
    return n;
}

void LabelTree::save(std::ofstream& out) {
    Log(CERR) << "Saving tree ...\n";

    int k = leaves.size();
    int t = nodes.size();

    out.write((char*)&k, sizeof(k));
    out.write((char*)&t, sizeof(t));

    for (size_t i = 0; i < t; ++i) {
        TreeNode* n = nodes[i];
        out.write((char*)&n->index, sizeof(n->index));
        out.write((char*)&n->label, sizeof(n->label));
    }

    int rootN = root->index;
    out.write((char*)&rootN, sizeof(rootN));

    for (size_t i = 0; i < t; ++i) {
        TreeNode* n = nodes[i];

        int parentN;
        if (n->parent)
            parentN = n->parent->index;
        else
            parentN = -1;

        out.write((char*)&parentN, sizeof(parentN));
    }
}

void LabelTree::load(std::ifstream& in) {
    clear();
    Log(CERR) << "Loading tree ...\n";

    int k, t;
    in.read((char*)&k, sizeof(k));
    in.read((char*)&t, sizeof(t));
    for (size_t i = 0; i < t; ++i) {
        TreeNode* n = new TreeNode();
        in.read((char*)&n->index, sizeof(n->index));
        in.read((char*)&n->label, sizeof(n->label));

        nodes.push_back(n);
        if (n->label >= 0) leaves[n->label] = n;
    }

    int rootN;
    in.read((char*)&rootN, sizeof(rootN));
    root = nodes[rootN];

    for (size_t i = 0; i < t; ++i) {
        TreeNode* n = nodes[i];

        int parentN;
        in.read((char*)&parentN, sizeof(parentN));
        if (parentN >= 0) {
            nodes[parentN]->children.push_back(n);
            n->parent = nodes[parentN];
        }
    }

    Log(CERR, 2) << "Loaded nodes: " << nodes.size() << ", leaves: " << leaves.size() << "\n";
}

void LabelTree::printTree(TreeNode* rootNode, bool printNodes) {
    Log(CERR) << "Tree:";
    if (rootNode == nullptr) rootNode = root;

    UnorderedSet<TreeNode*> nSet;
    std::queue<TreeNode*> nQueue;
    nQueue.push(rootNode);
    nSet.insert(rootNode);
    int depth = 0;
    if(printNodes) Log(CERR) << "\nDepth " << depth << ":";

    while (!nQueue.empty()) {
        TreeNode* n = nQueue.front();
        nQueue.pop();

        if (nSet.count(n->parent)) {
            if(printNodes) Log(CERR) << "\nDepth " << ++depth << ":";
            else Log(CERR) << " " << nSet.size();
            nSet.clear();
        }

        nSet.insert(n);
        if(printNodes) {
            Log(CERR) << " " << n->index;
            if (n->parent) Log(CERR) << "(" << n->parent->index << ")";
            if (n->label >= 0) Log(CERR) << "<" << n->label << ">";
        }
        for (auto c : n->children) nQueue.push(c);
    }

    if(!printNodes) Log(CERR) << " " << nSet.size();
    Log(CERR) << "\n";
}

int LabelTree::getNumberOfLeaves(TreeNode* rootNode) {
    if (rootNode == nullptr) // Root node
        return leaves.size();

    int lCount = 0;
    std::queue<TreeNode*> nQueue;
    nQueue.push(rootNode);

    while (!nQueue.empty()) {
        TreeNode* n = nQueue.front();
        nQueue.pop();

        if (n->label >= 0) ++lCount;
        for (auto c : n->children) nQueue.push(c);
    }

    return lCount;
}

void LabelTree::setLabel(TreeNode* n, int label) {
    n->label = label;
    if (label >= 0) {
        auto f = leaves.find(label);
        if (f != leaves.end()) f->second->label = -1;
        leaves[n->label] = n;
    }
}

int LabelTree::getTreeDepth(TreeNode* rootNode) {
    if (rootNode == nullptr) // Root node
        rootNode = root;

    int maxDepth = 1;
    std::queue<std::pair<int, TreeNode*>> nQueue;
    nQueue.push({1, root});

    while (!nQueue.empty()) {
        auto n = nQueue.front(); // Current node
        nQueue.pop();

        if (n.first > maxDepth) maxDepth = n.first;

        for (const auto& child : n.second->children) nQueue.push({n.first + 1, child});
    }

    return maxDepth;
}

int LabelTree::getNodeDepth(TreeNode* n) {
    uint32_t nDepth = 1;

    while (n != root) {
        n = n->parent;
        ++nDepth;
    }

    return nDepth;
}

void LabelTree::moveSubtree(TreeNode* oldParent, TreeNode* newParent) {
    if (oldParent->children.size()) {
        for (auto child : oldParent->children) setParent(child, newParent);
        oldParent->children.clear();
    } else
        setLabel(newParent, oldParent->label);

    setParent(newParent, oldParent);
}

int LabelTree::distanceBetweenNodes(TreeNode* n1, TreeNode* n2) {
    UnorderedMap<TreeNode*, int> path1;

    int i = 0;
    TreeNode* n = n1;
    while (n != nullptr) {
        path1.insert({n, i++});
        n = n->parent;
    }

    i = 0;
    n = n2;
    while (n != nullptr) {
        auto fn = path1.find(n);
        if (fn != path1.end()) return fn->second + i;
        n = n->parent;
        ++i;
    }

    return INT_MAX;
}