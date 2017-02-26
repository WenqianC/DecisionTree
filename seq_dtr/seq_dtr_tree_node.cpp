//
//  seq_dtr_tree_node.cpp
//  Classifer_RF
//
//  Created by jimmy on 2017-02-25.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "seq_dtr_tree_node.h"

SeqDTRTreeNode::~SeqDTRTreeNode()
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

void SeqDTRTreeNode::writeNode(FILE *pf, const NodePtr node)
{
    if (!node) {
        fprintf(pf, "#\n");
        return;
    }
    
    // write current node split parameter
    SeqDTRTreeNode::SplitParameter param = node->split_param_;
    fprintf(pf, "%2d\t %d\t %d\t %d\t %lf\n",
            node->depth_, (int)node->is_leaf_, param.split_time_step_, param.split_dim_, param.split_threshold_);
    
    
    // label mean
    fprintf(pf, "%d\n", (int)node->label_mean_.size());
    for (int i = 0; i<node->label_mean_.size(); i++) {
        fprintf(pf, "%lf ", node->label_mean_[i]);
    }
    fprintf(pf, "\n");
    
    // label standard deviation
    fprintf(pf, "%d\n", (int)node->label_std_.size());
    for (int i = 0; i<node->label_std_.size(); i++) {
        fprintf(pf, "%lf ", node->label_std_[i]);
    }
    fprintf(pf, "\n");
    
    SeqDTRTreeNode::writeNode(pf, node->left_child_);
    SeqDTRTreeNode::writeNode(pf, node->right_child_);
}


bool SeqDTRTreeNode::writeTree(const char *fileName,
                               const vector<unsigned int> & time_steps,
                               const vector<double> & weights,
                               NodePtr root)
{
    assert(root);
    assert(time_steps.size() == weights.size());
    FILE *pf = fopen(fileName, "w");
    if (!pf) {
        printf("can not open file %s\n", fileName);
        return false;
    }
    fprintf(pf, "%lu\n ", time_steps.size());
    for (int i = 0; i<time_steps.size(); i++) {
        fprintf(pf, "%d ", time_steps[i]);
    }
    fprintf(pf, "\n");
    
    fprintf(pf, "%lu\n ", weights.size());
    for (int i = 0; i<weights.size(); i++) {
        fprintf(pf, "%lf ", (double)weights[i]);
    }
    fprintf(pf, "\n");
    
    fprintf(pf, "depth\t isLeaf\t time_step\t dim\t threshold\t  mean stddev\n");
    
    SeqDTRTreeNode::writeNode(pf, root);
    fclose(pf);
    return true;
}

bool SeqDTRTreeNode::readTree(const char *fileName,
                              vector<unsigned int> & time_steps,
                              vector<double> & weights,
                              NodePtr & root)
{
    FILE *pf = fopen(fileName, "r");
    if (!pf) {
        printf("can not open file %s\n", fileName);
        return false;
    }
    // read time steps
    int num = 0;
    int ret = fscanf(pf, "%d", &num);
    assert(ret == 1);
    for (int i = 0; i<num; i++) {
        int t = 0;
        fscanf(pf, "%d", &t);
        time_steps.push_back(t);
    }
    
    ret = fscanf(pf, "%d", &num);
    assert(ret == 1);
    for (int i = 0; i<num; i++) {
        double v = 0;
        fscanf(pf, "%lf", &v);
        weights.push_back(v);
    }
    // remove two lines
    for (int i = 0; i<2; i++) {
        char line_buf[1024] = {NULL};
        fgets(line_buf, sizeof(line_buf), pf);
        printf("%s\n", line_buf);
    }
    
    SeqDTRTreeNode::readNode(pf, root);
    fclose(pf);
    return true;
}

void SeqDTRTreeNode::readNode(FILE *pf, NodePtr & node)
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
    int split_time_step = 0;
    int split_dim = 0;
    double split_threshold = 0.0;
    int ret_num = sscanf(lineBuf, "%d %d %d %d %lf",
                         &depth, &is_leaf, &split_time_step, &split_dim, &split_threshold);
    assert(ret_num == 5);
    node->depth_ = depth;
    node->is_leaf_ = is_leaf;
    node->split_param_.split_time_step_ = split_time_step;
    node->split_param_.split_dim_ = split_dim;
    node->split_param_.split_threshold_ = split_threshold;
    
    int label_dim = 0;
    ret_num = fscanf(pf, "%d", &label_dim);
    assert(ret_num == 1);
    Eigen::VectorXf mean = Eigen::VectorXf::Zero(label_dim);
    for (int i = 0; i<label_dim; i++) {
        double val = 0;
        ret_num = fscanf(pf, "%lf", &val);
        assert(ret_num);
        mean[i] = val;
    }
    node->label_mean_ = mean;
    
    // label standard deviation
    ret_num = fscanf(pf, "%d", &label_dim);
    assert(ret_num == 1);
    Eigen::VectorXf stddev = Eigen::VectorXf::Zero(label_dim);
    for (int i = 0; i<label_dim; i++) {
        double val = 0;
        ret_num = fscanf(pf, "%lf", &val);
        assert(ret_num);
        stddev[i] = val;
    }
    node->label_std_ = stddev;
    
    // remove '\n' at the end of the line
    char dummy_line_buf[1024] = {NULL};
    fgets(dummy_line_buf, sizeof(dummy_line_buf), pf);
    
    SeqDTRTreeNode::readNode(pf, node->left_child_);
    SeqDTRTreeNode::readNode(pf, node->right_child_);
}



