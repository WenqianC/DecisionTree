//
//  DTCNode.cpp
//  Classifer_RF
//
//  Created by jimmy on 2016-07-26.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#include "DTCNode.h"


static void write_DTCNode(FILE *pf, DTCNode * node)
{
    if (!node) {
        fprintf(pf, "#\n");
        return;
    }
    // write current node
    DTCSplitParameter param = node->split_param_;
    fprintf(pf, "%2d\t %d\t %6d\t\t %lf\t %.2f\t\t %d\n",
            node->depth_, (int)node->is_leaf_,  param.split_dim_, param.split_threshold_, node->sample_percentage_, node->sample_num_);
    
    if (node->is_leaf_) {
        fprintf(pf, "%d\n", (int)node->prob_.size());
        for (int i = 0; i<node->prob_.size(); i++) {
            fprintf(pf, "%lf ", node->prob_[i]);
        }
        fprintf(pf, "\n");
    }
    
    write_DTCNode(pf, node->left_child_);
    write_DTCNode(pf, node->right_child_);
}


bool DTCNode::writeTree(const char *fileName, DTCNode * root)
{
    assert(root);
    FILE *pf = fopen(fileName, "w");
    if (!pf) {
        printf("Error: can not open file %s\n", fileName);
        return false;
    }
    fprintf(pf, "depth\t isLeaf\t splitDim\t threshold\t percentage\t num\t probability\n");
    write_DTCNode(pf, root);
    fclose(pf);
    return true;
}

static void read_DTCNode(FILE *pf, DTCNode * & node)
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
    node = new DTCNode(0);
    assert(node);
    int depth = 0;
    int is_leaf = 0;
    int split_dim = 0;
    double split_threshold = 0.0;
    int sample_num = 0;
    double sample_percentage = 0.0;
    
    int ret_num = sscanf(lineBuf, "%d %d %d %lf %lf %d",
                         &depth, &is_leaf, &split_dim, &split_threshold, &sample_percentage, &sample_num);
    assert(ret_num == 6);
    
    node->depth_ = depth;
    node->is_leaf_ = is_leaf;
    node->sample_num_ = sample_num;
    node->sample_percentage_ = sample_percentage;
    
    DTCSplitParameter param;
    param.split_dim_ = split_dim;
    param.split_threshold_ = split_threshold;
    node->split_param_ = param;
    
    if (is_leaf) {
        int label_dim = 0;
        ret_num = fscanf(pf, "%d", &label_dim);
        assert(ret_num == 1);
        Eigen::VectorXd prob = Eigen::VectorXd::Zero(label_dim);
        for (int i = 0; i<label_dim; i++) {
            double val = 0;
            ret_num = fscanf(pf, "%lf", &val);
            assert(ret_num);
            prob[i] = val;
        }
        // remove '\n' at the end of the line
        char dummy_line_buf[1024] = {NULL};
        fgets(dummy_line_buf, sizeof(dummy_line_buf), pf);
        node->prob_ = prob;        
    }
    read_DTCNode(pf, node->left_child_);
    read_DTCNode(pf, node->right_child_);
}

bool DTCNode::readTree(const char *fileName, DTCNode * & root)
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
    read_DTCNode(pf, root);
    fclose(pf);
    return true;
}