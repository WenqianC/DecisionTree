//
//  DTRegressor.cpp
//  Classifer_RF
//
//  Created by jimmy on 2016-10-04.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#include "DTRegressor.h"

bool DTRegressor::predict(const Eigen::VectorXd & feature,
                          Eigen::VectorXd & pred) const
{
    assert(trees_.size() > 0);
    pred = Eigen::VectorXd::Zero(tree_param_.label_dim_);
    
    // average predictions from all trees
    int pred_num = 0;
    for (int i = 0; i<trees_.size(); i++) {
        Eigen::VectorXd cur_pred;
        bool is_pred = trees_[i]->predict(feature, cur_pred);
        if (is_pred) {
            pred += cur_pred;
            pred_num++;
        }
    }
    if (pred_num == 0) {
        return false;
    }
    pred /= pred_num;
    return true;
}