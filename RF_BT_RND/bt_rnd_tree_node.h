//
//  bt_rnd_tree_node.h
//  RGBD_RF
//
//  Created by jimmy on 2017-01-18.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __RGBD_RF__bt_rnd_tree_node__
#define __RGBD_RF__bt_rnd_tree_node__

// backtracking random tree node
#include <stdio.h>
#include <vector>
#include <Eigen/Dense>
#include "bt_rnd_util.h"


using std::vector;
using Eigen::VectorXf;

class BTRNDTreeNode
{
    friend class BTRNDTree;
private:
    
    typedef BTRNDTreeNode* NodePtr;
    typedef RandomSplitParameter SplitParameter;

    BTRNDTreeNode *left_child_;
    BTRNDTreeNode *right_child_;
    int depth_;
    bool is_leaf_;
    
    // split parameter
    SplitParameter split_param_;
    
    // leaf node property
    VectorXf label_mean_;      // label
    VectorXf label_stddev_;
    VectorXf feat_mean_;
    int index_;               // leaf node index, in-order traversal, leaf node only
    
public:
    BTRNDTreeNode(int depth)
    {
        left_child_ = NULL;
        right_child_ = NULL;
        depth_   = depth;
        is_leaf_ = false;
        index_ = -1;
    }
    ~BTRNDTreeNode();
    
    static bool writeTree(const char *fileName, const NodePtr root, const int leafNodeNum);
    static bool readTree(const char *fileName, NodePtr & root, int &leafNodeNum);
    
private:
    static void writeNode(FILE *pf, const NodePtr node);
    static void readNode(FILE *pf, NodePtr & node);
};




#endif /* defined(__RGBD_RF__bt_rnd_tree_node__) */
