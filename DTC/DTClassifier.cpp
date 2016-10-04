//
//  DTClassifier.cpp
//  Classifer_RF
//
//  Created by jimmy on 2016-09-20.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#include "DTClassifier.h"


bool DTClassifer::predict(const Eigen::VectorXd & feature,
                          Eigen::VectorXd & prob) const
{
    assert(trees_.size() > 0);
    
    const DTCTreeParameter param = trees_[0]->getTreeParameter();
    prob = Eigen::VectorXd::Zero(param.category_num_);
    
    for (int i = 0; i<trees_.size(); i++) {
        Eigen::VectorXd p;
        trees_[i]->predict(feature, p);
        prob += p;
    }
    prob /= trees_.size();
    
    return true;
}