/**
 * Copyright (c) 2018 by Marek Wydmuch, Kalina Jasinska, Robert Istvan Busa-Fekete
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
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

#include "data_reader.h"
#include "threads.h"
#include "tree.h"

Tree::Tree() {
    online = false;
    nextToExpand = 0;
}

Tree::~Tree() {
    for (auto n : nodes) delete n;
}

void Tree::buildTreeStructure(int labelCount, Args& args) {
    // Create a tree structure
    std::cerr << "Building tree ...\n";

    if (args.treeType == completeInOrder)
        buildCompleteTree(labelCount, false, args);
    else if (args.treeType == completeRandom)
        buildCompleteTree(labelCount, true, args);
    else if (args.treeType == balancedInOrder)
        buildBalancedTree(labelCount, false, args);
    else if (args.treeType == balancedRandom)
        buildBalancedTree(labelCount, true, args);
    else if (args.treeType == huffman || args.treeType == hierarchicalKMeans)
        std::cerr << "This tree type is not supported for this model type\n";
    else if (args.treeType == onlineBalanced || args.treeType == onlineComplete || args.treeType == onlineRandom ||
             args.treeType == onlineBottomUp) {
        online = true;
    } else if (args.treeType != custom) {
        std::cerr << "Unknown tree type\n";
        exit(1);
    }
}

void Tree::buildTreeStructure(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args) {
    // Load tree structure from file
    if (!args.treeStructure.empty()) loadTreeStructure(args.treeStructure);

    // Create a tree structure
    std::cerr << "Building tree ...\n";

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
        computeLabelsFeaturesMatrix(labelsFeatures, labels, features, args.threads, args.norm,
                                    args.kMeansWeightedFeatures);
        // labelsFeatures.save(joinPath(args.model, "lf_mat.bin"));
        buildKMeansTree(labelsFeatures, args);
        exit(0);
    } else if (args.treeType == onlineBalanced || args.treeType == onlineComplete || args.treeType == onlineRandom) {
        buildOnlineTree(labels, features, args);
    } else if (args.treeType != custom) {
        throw std::invalid_argument("buildTreeStructure: Unknown tree type");
    }

    // Check tree
    assert(k == leaves.size());
    assert(t == nodes.size());
}

TreeNodePartition Tree::buildKMeansTreeThread(TreeNodePartition nPart, SRMatrix<Feature>& labelsFeatures, Args& args,
                                              int seed) {
    kMeans(nPart.partition, labelsFeatures, args.arity, args.kMeansEps, args.kMeansBalanced, seed);
    return nPart;
}

void Tree::buildKMeansTree(SRMatrix<Feature>& labelsFeatures, Args& args) {
    std::cerr << "Hierarchical K-Means clustering in " << args.threads << " threads ...\n";

    root = createTreeNode();
    k = labelsFeatures.rows();

    long seed = args.getSeed();
    std::default_random_engine rng(seed);
    std::uniform_int_distribution<int> kMeansSeeder(0, INT_MAX);

    auto partition = new std::vector<Assignation>(k);
    for (int i = 0; i < k; ++i) (*partition)[i].index = i;

    // Run clustering in parallel
    ThreadPool tPool(args.threads);
    std::vector<std::future<TreeNodePartition>> results;

    TreeNodePartition rootPart = {root, partition};
    results.emplace_back(
        tPool.enqueue(buildKMeansTreeThread, rootPart, std::ref(labelsFeatures), std::ref(args), kMeansSeeder(rng)));

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
                results.emplace_back(tPool.enqueue(buildKMeansTreeThread, childPart, std::ref(labelsFeatures),
                                                   std::ref(args), kMeansSeeder(rng)));
            }
        }

        delete nPart.partition;
    }

    t = nodes.size();
    assert(k == leaves.size());
    std::cerr << "  Nodes: " << nodes.size() << ", leaves: " << leaves.size() << "\n";
}

void Tree::buildHuffmanTree(SRMatrix<Label>& labels, Args& args) {
    std::cerr << "Building Huffman Tree ...\n";

    k = labels.cols();

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
        double aggregatedProb = 0;
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

    t = nodes.size(); // size of the tree
    std::cerr << "  Nodes: " << nodes.size() << ", leaves: " << leaves.size() << ", arity: " << args.arity << "\n";
}

void Tree::buildBalancedTree(int labelCount, bool randomizeOrder, Args& args) {
    std::cerr << "Building balanced Tree ...\n";

    root = createTreeNode();
    k = labelCount;
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

    t = nodes.size();
    assert(k == leaves.size());
    std::cerr << "  Nodes: " << nodes.size() << ", leaves: " << leaves.size() << "\n";
}

void Tree::buildCompleteTree(int labelCount, bool randomizeOrder, Args& args) {
    std::cerr << "Building complete Tree ...\n";

    std::default_random_engine rng(args.getSeed());

    k = labelCount;
    t = static_cast<int>(ceil(static_cast<double>(args.arity * k - 1) / (args.arity - 1)));

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

        TreeNode* parent = nodes[static_cast<int>(floor(static_cast<double>(i - 1) / args.arity))];
        createTreeNode(parent, label);
    }

    std::cerr << "  Nodes: " << nodes.size() << ", leaves: " << leaves.size() << ", arity: " << args.arity << "\n";
}

void Tree::buildOnlineTree(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args) {
    std::cerr << "Building online tree ...\n";

    int rows = features.rows();
    for (int r = 0; r < rows; ++r) {
        printProgress(r, rows);

        auto rSize = labels.size(r);
        auto rLabels = labels[r];

        // Check row
        for (int i = 0; i < rSize; ++i) {
            if (!leaves.count(rLabels[i])) {

                int newLabel = rLabels[i];

                if (nodes.empty()) { // Empty tree
                    root = createTreeNode(nullptr, newLabel);
                    nextSubtree = root;
                    continue;
                }

                TreeNode* toExpand = root;

                // Select node based on policy
                if (args.treeType == onlineBalanced) { // Balanced policy
                    while (toExpand->children.size() >= args.arity) {
                        toExpand = toExpand->children[toExpand->nextToExpand++ % toExpand->children.size()];
                    }
                } else if (args.treeType == onlineComplete) { // Complete
                    if (nodes[nextToExpand]->children.size() >= args.arity) ++nextToExpand;
                    toExpand = nodes[nextToExpand];
                } else if (args.treeType == onlineRandom) { // Random policy
                    std::default_random_engine rng(args.getSeed());
                    std::uniform_int_distribution<uint32_t> dist(0, args.arity - 1);
                    while (toExpand->children.size() == args.arity) toExpand = toExpand->children[dist(rng)];
                }

                // Expand selected node
                if (toExpand->children.empty()) TreeNode* parentLabelNode = createTreeNode(toExpand, toExpand->label);
                TreeNode* newLabelNode = createTreeNode(toExpand, newLabel);
            }
        }
    }

    t = nodes.size(); // size of the tree
    std::cerr << "  Nodes: " << nodes.size() << ", leaves: " << leaves.size() << ", arity: " << args.arity << "\n";
}

void Tree::loadTreeStructure(std::string file) {
    std::cerr << "Loading Tree structure from: " << file << "...\n";

    std::ifstream in(file);
    in >> k >> t;

    if (k >= t) throw std::invalid_argument("Specified number of labels is higher then specified number of nodes!");

    root = createTreeNode();
    for (int i = 1; i < t; ++i) createTreeNode();

    std::cerr << "  Header: nodes: " << t << ", leaves: " << k << "\n";

    std::string line;
    while (std::getline(in, line)) {
        if (!line.length()) continue;

        int parent, child, label = -1;
        std::string sLabel;

        std::istringstream lineISS(line);
        lineISS >> parent >> child >> sLabel;
        if (sLabel.size()) label = std::stoi(sLabel);

        if (child >= t) throw std::invalid_argument("Node index is higher then specified number of nodes!");
        if (parent >= t) throw std::invalid_argument("Parent index is higher then specified number of nodes!");
        if (label >= k) throw std::invalid_argument("Label index is higher then specified number of labels!");

        if (parent == -1) {
            root = nodes[child];
            continue;
        }

        TreeNode* parentN = nodes[parent];
        TreeNode* childN = nodes[child];
        parentN->children.push_back(childN);
        childN->parent = parentN;

        if (label >= 0) {
            assert(leaves.count(label) == 0);
            assert(label < k);
            childN->label = label;
            leaves[childN->label] = childN;
        }
    }
    in.close();

    // Additional validation of a tree
    for (const auto& n : nodes) {
        if (n->parent == nullptr && n != root)
            throw std::invalid_argument("A node without parent, that is not a tree root exists!");
        if (n->children.size() == 0 && n->label < 0)
            throw std::invalid_argument("An internal node without children exists!");
    }

    assert(nodes.size() == t);
    assert(leaves.size() == k);
    std::cerr << "  Loaded: nodes: " << nodes.size() << ", leaves: " << leaves.size() << "\n";
}

void Tree::saveTreeStructure(std::string file) {
    std::cerr << "Saving Tree structure to: " << file << "...\n";

    std::ofstream out(file);
    out << k << " " << t << "\n";
    for (auto& n : nodes) {
        if (n->parent != nullptr) {
            out << n->parent->index;
            // else out << -1
            out << " " << n->index << " ";
            if (n->label >= 0) out << n->label;
            // else out << -1;
            out << "\n";
        }
    }
    out.close();
}

TreeNode* Tree::createTreeNode(TreeNode* parent, int label) {
    TreeNode* n = new TreeNode();
    n->index = nodes.size();
    nodes.push_back(n);
    setLabel(n, label);
    setParent(n, parent);
    n->nextToExpand = 0;
    n->subtreeDepth = 1;
    return n;
}

void Tree::save(std::ostream& out) {
    std::cerr << "Saving tree ...\n";

    out.write((char*)&k, sizeof(k));

    t = nodes.size();
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

void Tree::load(std::istream& in) {
    std::cerr << "Loading tree ...\n";

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

    std::cerr << "  Nodes: " << nodes.size() << ", leaves: " << leaves.size() << "\n";
}

void Tree::printTree(TreeNode* rootNode) {
    std::cerr << "Tree:";
    if (rootNode == nullptr) rootNode = root;

    UnorderedSet<TreeNode*> nSet;
    std::queue<TreeNode*> nQueue;
    nQueue.push(rootNode);
    nSet.insert(rootNode);
    int depth = 0;
    std::cerr << "\nDepth " << depth << ":";

    while (!nQueue.empty()) {
        TreeNode* n = nQueue.front();
        nQueue.pop();

        if (nSet.count(n->parent)) {
            nSet.clear();
            std::cerr << "\nDepth " << ++depth << ":";
        }

        nSet.insert(n);
        std::cerr << " " << n->index;
        if (n->parent) std::cerr << "(" << n->parent->index << ")";
        if (n->label >= 0) std::cerr << "<" << n->label << ">";
        for (auto c : n->children) nQueue.push(c);
    }

    std::cerr << "\n";
}

int Tree::getNumberOfLeaves(TreeNode* rootNode) {
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

void Tree::setLabel(TreeNode* n, int label) {
    n->label = label;
    if (label >= 0) {
        auto f = leaves.find(label);
        if (f != leaves.end()) f->second->label = -1;
        leaves[n->label] = n;
    }
}

int Tree::getTreeDepth(TreeNode* rootNode) {
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

int Tree::getNodeDepth(TreeNode* n) {
    uint32_t nDepth = 1;

    while (n != root) {
        n = n->parent;
        ++nDepth;
    }

    return nDepth;
}

void Tree::moveSubtree(TreeNode* oldParent, TreeNode* newParent) {
    if (oldParent->children.size()) {
        for (auto child : oldParent->children) setParent(child, newParent);
        oldParent->children.clear();
    } else
        setLabel(newParent, oldParent->label);

    setParent(newParent, oldParent);
}

void Tree::expandTopDown(Label newLabel, std::vector<Base*>& bases, std::vector<Base*>& tmpBases, Args& args) {

    TreeNode* toExpand = root;

    // Select node based on policy
    if (args.treeType == onlineBalanced) { // Balanced policy
        while (toExpand->children.size() >= args.arity) {
            toExpand = toExpand->children[toExpand->nextToExpand++ % toExpand->children.size()];
        }
    } else if (args.treeType == onlineComplete) { // Complete
        if (nodes[nextToExpand]->children.size() >= args.arity) ++nextToExpand;
        toExpand = nodes[nextToExpand];
    } else if (args.treeType == onlineRandom) { // Random policy
        std::default_random_engine rng(args.getSeed());
        std::uniform_int_distribution<uint32_t> dist(0, args.arity - 1);
        while (toExpand->children.size() >= args.arity) toExpand = toExpand->children[dist(rng)];
    }

    // Expand selected node
    if (toExpand->children.size() == 0) {
        TreeNode* parentLabelNode = createTreeNode(toExpand, toExpand->label);
        bases.push_back(bases[toExpand->index]->copyInverted());
        tmpBases.push_back(tmpBases[toExpand->index]->copy());
    }

    TreeNode* newLabelNode = createTreeNode(toExpand, newLabel);
    bases.push_back(tmpBases[toExpand->index]->copyInverted());
    tmpBases.push_back(new Base());

    // Remove temporary classifier
    if (toExpand->children.size() == args.arity - 1) {
        delete tmpBases[toExpand->index];
        tmpBases[toExpand->index] = nullptr;
    }
}

void Tree::expandBottomUp(Label newLabel, std::vector<Base*>& bases, std::vector<Base*>& tmpBases, Args& args) {
    if (nextSubtree->children.size() < args.arity) {
        // Adding new child

        if (nextSubtree->label != -1) {
            TreeNode* labelChild = createTreeNode(nextSubtree, nextSubtree->label);
            bases.push_back(tmpBases[nextSubtree->index]->copyInverted());
            tmpBases.push_back(new Base());
            ++nextSubtree->subtreeDepth;
        }

        TreeNode* newChild = createTreeNode(nextSubtree, newLabel);
        bases.push_back(tmpBases[nextSubtree->index]->copy());
        tmpBases.push_back(new Base());

        if (nextSubtree->parent && nextSubtree->children.size() == args.arity - 1 &&
            nextSubtree->parent->subtreeDepth == nextSubtree->subtreeDepth + 1) {
            delete tmpBases[nextSubtree->index];
            tmpBases[nextSubtree->index] = nullptr;
        }

        if (nextSubtree->subtreeDepth > 2) nextSubtree = newChild;
    } else if (nextSubtree->parent && nextSubtree->parent->subtreeDepth == nextSubtree->subtreeDepth + 1) {
        // Moving up
        nextSubtree = nextSubtree->parent;
        expandBottomUp(newLabel, bases, tmpBases, args);
    } else {
        // Expanding subtree
        TreeNode* parentOfOldTree = createTreeNode();
        bases.push_back(tmpBases[nextSubtree->index]->copy());
        tmpBases.push_back(new Base());

        parentOfOldTree->subtreeDepth = nextSubtree->subtreeDepth;
        moveSubtree(nextSubtree, parentOfOldTree);
        ++nextSubtree->subtreeDepth;

        expandBottomUp(newLabel, bases, tmpBases, args);
    }
}

void Tree::expandTree(Label newLabel, std::vector<Base*>& bases, std::vector<Base*>& tmpBases, Args& args) {
    if (nodes.size() == 0) { // Empty tree
        root = createTreeNode(nullptr, newLabel);
        bases.emplace_back(new Base());
        tmpBases.emplace_back(new Base());
        nextSubtree = root;
    } else if (args.treeType == onlineBottomUp)
        expandBottomUp(newLabel, bases, tmpBases, args);
    else
        expandTopDown(newLabel, bases, tmpBases, args);
}

void Tree::populateNodeLabels() {
    for (auto& l : leaves) {
        TreeNode* n = l.second;
        while (n != nullptr) {
            n->labels.push_back(l.first);
            n = n->parent;
        }
    }
}

int Tree::distanceBetweenNodes(TreeNode* n1, TreeNode* n2) {
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