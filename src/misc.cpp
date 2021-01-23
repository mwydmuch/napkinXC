/*
 Copyright (c) 2018-2020 by Marek Wydmuch

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


#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include "misc.h"
#include "threads.h"

// Data utils
std::vector<Prediction> computeLabelsPriors(const SRMatrix<Label>& labels) {
    Log(CERR) << "Computing labels' prior probabilities ...\n";

    std::vector<Prediction> labelsProb;

    labelsProb.resize(labels.cols());
    for (int i = 0; i < labelsProb.size(); ++i) labelsProb[i] = {i, 0};

    int rows = labels.rows();
    for (int r = 0; r < rows; ++r) {
        printProgress(r, rows);
        int rLabels = labels.size(r);
        for (int i = 0; i < rLabels; ++i) ++labelsProb[labels[r][i]].value;
    }

    for (auto& p : labelsProb) p.value /= labels.rows();

    return labelsProb;
}

void computeLabelsFeaturesMatrixThread(std::vector<std::vector<Feature>>& labelsFeatures,
                                        std::vector<std::vector<int>>& labelsExamples,
                                        const SRMatrix<Label>& labels, const SRMatrix<Feature>& features,
                                        bool norm, bool weightedFeatures, int threadId, int threads){
    int size = labelsExamples.size();
    for (int l = threadId; l < size; l += threads) {
        if (threadId == 0) printProgress(l, size);
        UnorderedMap<int, double> lFeatures;

        for (const auto& e : labelsExamples[l]){
            auto f = features[e];
            if(f->index == 1) ++f; // Skip bias feature
            if(weightedFeatures) addVector(f, 1.0 / features.size(e), lFeatures);
            else addVector(f, 1.0, lFeatures);
        }

        for(auto& f : lFeatures)
            labelsFeatures[l].push_back({f.first, f.second});

        std::sort(labelsFeatures[l].begin(), labelsFeatures[l].end());
        if(norm) unitNorm(labelsFeatures[l]);
        else divVector(labelsFeatures[l], labelsExamples[l].size());
    }
}

void computeLabelsFeaturesMatrix(SRMatrix<Feature>& labelsFeatures, const SRMatrix<Label>& labels,
                                 const SRMatrix<Feature>& features, int threads, bool norm, bool weightedFeatures) {

    assert(features.rows() == labels.rows());
    Log(CERR) << "Computing labels' features matrix in " << threads << " threads ...\n";

    // Labels matrix transposed dot features matrix
    std::vector<std::vector<int>> labelsExamples(labels.cols());
    for(int i = 0; i < labels.rows(); ++i)
        for(int j = 0; j < labels.size(i); ++j)
            labelsExamples[labels[i][j]].push_back(i);

    std::vector<std::vector<Feature>> tmpLabelsFeatures(labels.cols());
    ThreadSet tSet;
    for (int t = 0; t < threads; ++t)
        tSet.add(computeLabelsFeaturesMatrixThread, std::ref(tmpLabelsFeatures), std::ref(labelsExamples),
                 std::ref(labels), std::ref(features), norm, weightedFeatures, t, threads);
    tSet.joinAll();

    for(auto& v : tmpLabelsFeatures)
        labelsFeatures.appendRow(v);
}

// Splits string
std::vector<std::string> split(std::string text, char d) {
    std::vector<std::string> tokens;
    const char* str = text.c_str();
    std::string strD = std::string("") + d;

    do {
        const char* begin = str;
        while (*str != d && *str) ++str;
        std::string token = std::string(begin, str);
        if (token.length() && token != strD) tokens.emplace_back(begin, str);
    } while (0 != *str++);

    return tokens;
}

std::string toLower(std::string text) {
    std::string lower = text;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    return lower;
}

std::string formatMem(size_t mem){
    // kilo, mega, giga, tera, peta, exa
    char units[7] = {' ', 'K', 'M', 'G', 'T', 'P', 'E'};
    double fMem = mem;
    int i = 0;
    while(fMem > 1024){
        fMem /= 1024;
        ++i;
    }
    mem = std::ceil(fMem);

    return "~" + std::to_string(mem) + units[i];
}

// Files utils
void FileHelper::saveToFile(std::string outfile) {
    std::ofstream out(outfile);
    save(out);
    out.close();
}

void FileHelper::loadFromFile(std::string infile) {
    checkFileName(infile);
    std::ifstream in(infile);
    load(in);
    in.close();
}

// Joins two paths
std::string joinPath(const std::string& path1, const std::string& path2) {
    char sep = '/';

    std::string joined = path1;
    if (path1[path1.size() - 1] != sep) joined += sep;
    if (path2[0] == sep)
        joined += path2.substr(1);
    else
        joined += path2;

    return (joined);
}

// Checks filename
void checkFileName(const std::string& filename, bool read) {
    bool valid;
    if (read) {
        std::ifstream in(filename);
        valid = in.good();
    } else {
        std::ofstream out(filename);
        valid = out.good();
    }
    if (!valid) throw std::invalid_argument("Invalid filename: \"" + filename + "\"!");
}

// Checks dirname
void checkDirName(const std::string& dirname) {
    std::string tmpFile = joinPath(dirname, ".checkTmp");
    std::ofstream out(tmpFile);
    if (!out.good()) throw std::invalid_argument("Invalid dirname: \"" + dirname + "\"!");
    std::remove(tmpFile.c_str());
}

// TODO improve this
// Run shell CMD
void shellCmd(const std::string& cmd) {
    const int cmdErr = std::system(cmd.c_str());
    if (-1 == cmdErr) { exit(1); }
}

// Create directory
void makeDir(const std::string& dirname) {
    std::string mkdirCmd = "mkdir -p " + dirname;
    shellCmd(mkdirCmd);
}

// Remove directory of file
void remove(const std::string& path) {
    std::string rmCmd = "rm -rf " + path;
    shellCmd(rmCmd);
}
