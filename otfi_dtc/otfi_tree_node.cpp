//
//  otfi_tree_node.cpp
//  SequentialRandomForest
//
//  Created by jimmy on 2017-07-16.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "otfi_tree_node.hpp"

OTFITreeNode::~OTFITreeNode()
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

bool OTFITreeNode::writeTree(const char *fileName,                               
                               NodePtr root)
{
    assert(root);
    FILE *pf = fopen(fileName, "w");
    if (!pf) {
        printf("can not open file %s\n", fileName);
        return false;
    }
    fprintf(pf, "depth\t isLeaf\t dim\t threshold\t lower_bound\t upper_bound\t  prob\n");
    OTFITreeNode::writeNode(pf, root);
    fclose(pf);
    return true;
}

void OTFITreeNode::writeNode(FILE *pf, const NodePtr node)
{
    if (!node) {
        fprintf(pf, "#\n");
        return;
    }
    // write current node split parameter
    SplitParameter param = node->split_param_;
    fprintf(pf, "%2d\t %d\t %d\t %lf %lf %lf\n",
            node->depth_, (int)node->is_leaf_, param.dim_,
            param.threshold_, param.lower_bound_, param.upper_bound_);
    
    if (node->is_leaf_) {
        // probability of each category in this node        
        for (int i = 0; i<node->prob_.size(); i++) {
            fprintf(pf, "%lf ", node->prob_[i]);
        }
        fprintf(pf, "\n");
    }
    
    
    OTFITreeNode::writeNode(pf, node->left_child_);
    OTFITreeNode::writeNode(pf, node->right_child_);
}

bool OTFITreeNode::readTree(const char *fileName,
                            const int category_num,                            
                              NodePtr & root)
{
    FILE *pf = fopen(fileName, "r");
    if (!pf) {
        printf("can not open file %s\n", fileName);
        return false;
    }    
    // remove one lines
    for (int i = 0; i<1; i++) {
        char line_buf[1024] = {NULL};
        fgets(line_buf, sizeof(line_buf), pf);
        printf("%s\n", line_buf);
    }
   
    OTFITreeNode::readNode(pf, category_num, root);
    fclose(pf);
    return true;
}

void OTFITreeNode::readNode(FILE *pf, const int category_num, NodePtr & node)
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
    double lower_bound = 0.0;
    double upper_bound = 0.0;
    int ret_num = sscanf(lineBuf, "%d %d %d %lf %lf %lf",
                         &depth, &is_leaf, &split_dim, &split_threshold,
                         &lower_bound, &upper_bound);
    assert(ret_num == 6);    
    node->depth_ = depth;
    node->is_leaf_ = is_leaf;    
    node->split_param_.dim_ = split_dim;
    node->split_param_.threshold_ = split_threshold;
    node->split_param_.lower_bound_ = lower_bound;
    node->split_param_.upper_bound_ = upper_bound;

    if (node->is_leaf_) {
        Eigen::VectorXf prob = Eigen::VectorXf::Zero(category_num);
        for (int i = 0; i<category_num; i++) {
            double val = 0;
            ret_num = fscanf(pf, "%lf", &val);
            assert(ret_num == 1);
            prob[i] = val;
        }
        node->prob_ = prob;
        
        // remove '\n' at the end of the line
        char dummy_line_buf[1024] = {NULL};
        fgets(dummy_line_buf, sizeof(dummy_line_buf), pf);
    }
    
    
    
    OTFITreeNode::readNode(pf, category_num, node->left_child_);
    OTFITreeNode::readNode(pf, category_num, node->right_child_);
}



