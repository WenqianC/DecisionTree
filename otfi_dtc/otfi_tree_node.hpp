//
//  otfi_tree_node.h
//  SequentialRandomForest
//
//  Created by jimmy on 2017-07-16.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __SequentialRandomForest__otfi_tree_node__
#define __SequentialRandomForest__otfi_tree_node__

#include <stdio.h>
#include <Eigen/Dense>
#include "otfi_util.hpp"

class OTFITree;

class OTFITreeNode
{
    friend OTFITree;
    
    typedef OTFITreeNode Node;
    typedef Node* NodePtr;
    typedef OTFISplitParameter SplitParameter;

    NodePtr left_child_;
    NodePtr right_child_;
    int depth_;
    bool is_leaf_;
    
    // split parameter
    SplitParameter split_param_;
    
    // only leaf node has probability
    Eigen::VectorXf prob_;         // label probability
    
public:
    OTFITreeNode(int depth)
    {
        left_child_ = NULL;
        right_child_ = NULL;
        depth_ = depth;
        is_leaf_ = false;
    }
    ~OTFITreeNode();
};

#endif /* defined(__SequentialRandomForest__otfi_tree_node__) */
