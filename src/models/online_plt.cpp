/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include "base.h"
#include "model.h"
#include "tree.h"


void OnlinePLT::update(Label*, Feature*, Args &args){
    std::unordered_set<TreeNode*> nPositive;
    std::unordered_set<TreeNode*> nNegative;

    getNodesToUpdate(nPositive, nNegative, labels.row(r), labels.size(r));

    for (const auto& n : nPositive){
        // Update positive base estimators
    }

    for (const auto& n : nNegative){
        // Update negative
    }
}
