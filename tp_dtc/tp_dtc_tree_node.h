//
//  tp_dtc_tree_node.h
//  Classifer_RF
//
//  Created by jimmy on 2017-02-19.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __Classifer_RF__tp_dtc_tree_node__
#define __Classifer_RF__tp_dtc_tree_node__

// trinary projection decision tree node

#include <stdio.h>
#include <vector>
#include <Eigen/Dense>
#include "tp_dtc_util.h"

using std::vector;
using Eigen::VectorXf;


class TPDTCTreeNode {
    typedef TPDTCTreeNode  Node;
    typedef TPDTCTreeNode* NodePtr;
    typedef TPDTCSplitParameter SplitParameter;
    
    friend class TPDTClassifier;
    
public:
    NodePtr left_child_;
    NodePtr right_child_;
    int depth_;
    bool is_leaf_;
    
    // split parameter
    SplitParameter split_param_;
    
    // only leaf node has probability
    VectorXf prob_;         // label probability
    
public:
    TPDTCTreeNode(int depth)
    {
        left_child_ = NULL;
        right_child_ = NULL;
        depth_ = depth;
        is_leaf_ = false;
    }
    ~TPDTCTreeNode();
    
    static bool writeTree(const char *fileName, NodePtr root);
    static bool readTree(const char *fileName, NodePtr & root);
    
private:
    static void writeNode(FILE *pf, const NodePtr node);
    static void readNode(FILE *pf, NodePtr & node);
    
};


#endif /* defined(__Classifer_RF__tp_dtc_tree_node__) */
