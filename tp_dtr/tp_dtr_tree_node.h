//
//  tp_dtr_tree_node.h
//  Classifer_RF
//
//  Created by jimmy on 2017-02-16.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __Classifer_RF__tp_dtr_tree_node__
#define __Classifer_RF__tp_dtr_tree_node__

// trinary projection decision tree regression
#include <stdio.h>
#include <vector>
#include <Eigen/Dense>
#include "tp_dtr_util.h"

using std::vector;
using Eigen::VectorXf;

class TPDTRTreeNode {
    typedef TPDTRTreeNode  Node;
    typedef TPDTRTreeNode* NodePtr;
    typedef TPDTRSplitParameter SplitParameter;
    
public:
    TPDTRTreeNode *left_child_;
    TPDTRTreeNode *right_child_;
    int depth_;
    bool is_leaf_;
    
    // split parameter
    SplitParameter split_param_;
    int sample_num_;
    double sample_percentage_;  // sample percentage of parent node
    
    // only leaf node has probability
    VectorXf label_mean_;  // label
    VectorXf label_stddev_;
    
public:
    TPDTRTreeNode(int depth)
    {
        left_child_ = NULL;
        right_child_ = NULL;
        depth_ = depth;
        is_leaf_ = false;
        sample_num_ = 0;
        sample_percentage_ = 0.0;
    }
    ~TPDTRTreeNode();
    
    static bool writeTree(const char *fileName, NodePtr root);
    static bool readTree(const char *fileName, NodePtr & root);
    
private:
    static void writeNode(FILE *pf, const NodePtr node);
    
};


#endif /* defined(__Classifer_RF__tp_dtr_tree_node__) */
