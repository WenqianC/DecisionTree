//
//  DTCNode.h
//  Classifer_RF
//
//  Created by jimmy on 2016-07-26.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef __Classifer_RF__DTCNode__
#define __Classifer_RF__DTCNode__

#include <stdio.h>
#include <Eigen/Dense>
#include "DTCUtil.h"

// decision tree classifer node
using Eigen::VectorXd;
class DTCNode
{
public:
    DTCNode *left_child_;
    DTCNode *right_child_;
    int depth_;
    bool is_leaf_;
    
    // split parameter
    DTCSplitParameter split_param_;
    int sample_num_;
    double sample_percentage_;  // sample percentage of parent node
    
    // only leaf node has probability
    VectorXd prob_;  // label probability
    
public:
    DTCNode(int depth)
    {
        left_child_ = NULL;
        right_child_ = NULL;
        depth_ = depth;
        is_leaf_ = false;
        
        sample_num_ = 0;
        sample_percentage_ = 0.0;
    }
    
    static bool writeTree(const char *fileName, DTCNode * root);
    static bool readTree(const char *fileName, DTCNode * & root);
};

#endif /* defined(__Classifer_RF__DTCNode__) */
