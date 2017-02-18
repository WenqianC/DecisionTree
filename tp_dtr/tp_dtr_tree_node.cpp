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
    fprintf(pf, "%2d \t\t", (int)param.split_weight_.size());
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
    
    fprintf(pf, "depth\t isLeaf\t dim\t threshold\t  channel_weights\t mean\t stddev\n");
    
    TPDTRTreeNode::writeNode(pf, root);
    fclose(pf);
    return true;
}

bool TPDTRTreeNode::readTree(const char *fileName, NodePtr & root)
{
    FILE *pf = fopen(fileName, "r");
    if (!pf) {
        printf("can not open file %s\n", fileName);
        return false;
    }
    //read first line
    char line_buf[1024] = {NULL};
    fgets(line_buf, sizeof(line_buf), pf);
    printf("%s\n", line_buf);
    TPDTRTreeNode::readNode(pf, root);
    fclose(pf);
    return true;
}

void TPDTRTreeNode::readNode(FILE *pf, NodePtr & node)
{
    char lineBuf[1024] = {NULL};
    char *ret = fgets(lineBuf, sizeof(lineBuf), pf);
    if (!ret) {
        node = NULL;
        return;
    }
    if (lineBuf[0] == '#') {
        // empty node
        node = NULL;
        return;
    }
    
    // read node parameters
    node = new Node(0);
    assert(node);
    int depth = 0;
    int is_leaf = 0;
    int split_dim = 0;
    double split_threshold = 0.0;
    int ret_num = sscanf(lineBuf, "%d %d %d %lf",
                         &depth, &is_leaf, &split_dim, &split_threshold);
    assert(ret_num == 4);
    node->depth_ = depth;
    node->is_leaf_ = is_leaf;
    node->split_param_.split_dim_ = split_dim;
    node->split_param_.split_threshold_ = split_threshold;
    node->split_param_.split_weight_.clear();
    
    int channels = 0;
    ret_num = fscanf(pf, "%d", &channels);
    assert(ret_num == 1);
    for (int c = 0; c < channels; c++) {
        int v = 0;
        ret_num = fscanf(pf, "%d", &v);
        assert(ret_num == 1);
        node->split_param_.split_weight_.push_back(v);
    }    
    
    if (is_leaf) {
        int label_dim = 0;
        ret_num = fscanf(pf, "%d", &label_dim);
        assert(ret_num == 1);
        Eigen::VectorXf mean = Eigen::VectorXf::Zero(label_dim);
        Eigen::VectorXf stddev = Eigen::VectorXf::Zero(label_dim);
        for (int i = 0; i<label_dim; i++) {
            double val = 0;
            ret_num = fscanf(pf, "%lf", &val);
            assert(ret_num);
            mean[i] = val;
        }
        for (int i = 0; i<label_dim; i++) {
            double val = 0;
            ret_num = fscanf(pf, "%lf", &val);
            assert(ret_num);
            stddev[i] = val;
        }
        // remove '\n' at the end of the line
        //char dummy_line_buf[1024] = {NULL};
        //fgets(dummy_line_buf, sizeof(dummy_line_buf), pf);
        node->label_mean_ = mean;
        node->label_stddev_ = stddev;        
    }
    // remove '\n' at the end of the line
    char dummy_line_buf[1024] = {NULL};
    fgets(dummy_line_buf, sizeof(dummy_line_buf), pf);
    
    TPDTRTreeNode::readNode(pf, node->left_child_);
    TPDTRTreeNode::readNode(pf, node->right_child_);
}
