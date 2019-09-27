/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include "base.h"
#include "model.h"
#include "tree.h"


class OnlinePLT: public PLT, public OnlineModel{
public:
    OnlinePLT();
    ~OnlinePLT() override;

    void update(Label*, Feature*, Args &args) override;
    void save(Args &args, std::string output) override;
};
