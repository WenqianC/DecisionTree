//
//  DTRNode.h
//  Classifer_RF
//
//  Created by jimmy on 2016-10-03.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef __Classifer_RF__DTRNode__
#define __Classifer_RF__DTRNode__

// decision tree regression node

#include <stdio.h>
#include <vector>
#include <Eigen/Dense>
#include "DTRUtil.h"

using std::vector;
using Eigen::VectorXd;

class DTRNode
{
public:
    DTRNode *left_child_;
    DTRNode *right_child_;
    int depth_;
    bool is_leaf_;
    
    // split parameter
    DTRSplitParameter split_param_;
    
    // only leaf node has probability
    VectorXd mean_;  // label
    VectorXd stddev_;
    
public:
    DTRNode(int depth)
    {
        left_child_ = NULL;
        right_child_ = NULL;
        depth_ = depth;
        is_leaf_ = false;
    }
};



#endif /* defined(__Classifer_RF__DTRNode__) */
