//
//  seq_dtr_tree_node.h
//  Classifer_RF
//
//  Created by jimmy on 2017-02-25.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __Classifer_RF__seq_dtr_tree_node__
#define __Classifer_RF__seq_dtr_tree_node__

// sequential decision tree (regressor) node
#include <stdio.h>

#include <vector>
#include <Eigen/Dense>
#include "seq_dtr_util.h"

using std::vector;
using Eigen::VectorXf;


class SeqDTRTreeNode {
    typedef SeqDTRTreeNode  Node;
    typedef SeqDTRTreeNode* NodePtr;
    typedef SeqDTRSplitParameter SplitParameter;
    
    friend class SeqDTClassifier;
    
public:
    NodePtr left_child_;
    NodePtr right_child_;
    int depth_;
    bool is_leaf_;
    
    // split parameter
    SplitParameter split_param_;
    
    // only leaf node has mean of labels
    VectorXf label_mean_;    // label
    VectorXf label_std_;
    
public:
    SeqDTRTreeNode(int depth)
    {
        left_child_ = NULL;
        right_child_ = NULL;
        depth_ = depth;
        is_leaf_ = false;
    }
    ~SeqDTRTreeNode();
    
    
    static bool writeTree(const char *fileName,
                          const vector<unsigned int> & time_steps,
                          const vector<double> & weights,
                          NodePtr root);
    
    static bool readTree(const char *fileName,
                         vector<unsigned int> & time_steps,
                         vector<double> & weights,
                         NodePtr & root);
    
private:
    static void writeNode(FILE *pf, const NodePtr node);
    static void readNode(FILE *pf, NodePtr & node);
    
};



#endif /* defined(__Classifer_RF__seq_dtr_tree_node__) */
