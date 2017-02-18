//
//  tp_dtr_tree_node.cpp
//  Classifer_RF
//
//  Created by jimmy on 2017-02-16.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "tp_dtr_tree_node.h"

TPDTRTreeNode::~TPDTRTreeNode()
{
    if (left_child_) {
        delete left_child_;
        left_child_ = NULL;
    }
    if (right_child_) {
        delete right_child_;
        right_child_ = NULL;
    }
}

void TPDTRTreeNode::writeNode(FILE *pf, const NodePtr node)
{
    if (!node) {
        fprintf(pf, "#\n");
        return;
    }
    
    // write current node split parameter
    TPDTRTreeNode::SplitParameter param = node->split_param_;
    fprintf(pf, "%2d\t %d\t %d\t %lf\n",
            node->depth_, (int)node->is_leaf_, param.split_dim_, param.split_threshold_);
    // channel weight or feature combination
    fprintf(pf, "%lu\n", param.split_weight_.size());
    for (int i = 0; i<param.split_weight_.size(); i++) {
        if (i != param.split_weight_.size() - 1) {
            fprintf(pf, "%d\t", param.split_weight_[i]);
        }
        else {
            fprintf(pf, "%d\n", param.split_weight_[i]);
        }
    }
    
    if (node->is_leaf_) {
        // leaf index and mean size
        fprintf(pf, "%d\n", (int)node->label_mean_.size());
        for (int i = 0; i<node->label_mean_.size(); i++) {
            fprintf(pf, "%lf ", node->label_mean_[i]);
        }
        fprintf(pf, "\n");
        for (int i = 0; i<node->label_stddev_.size(); i++) {
            fprintf(pf, "%lf ", node->label_stddev_[i]);
        }
        fprintf(pf, "\n");
    }
    
    TPDTRTreeNode::writeNode(pf, node->left_child_);
    TPDTRTreeNode::writeNode(pf, node->right_child_);
}


bool TPDTRTreeNode::writeTree(const char *fileName, NodePtr root)
{
    assert(root);
    FILE *pf = fopen(fileName, "w");
    if (!pf) {
        printf("can not open file %s\n", fileName);
        return false;
    }
    
    fprintf(pf, "depth\t isLeaf\t dim\t weight\t threshold\t mean\t stddev\n");
    
    TPDTRTreeNode::writeNode(pf, root);
    fclose(pf);
    return true;
}

bool TPDTRTreeNode::readTree(const char *fileName, NodePtr & root)
{
    return true;
}