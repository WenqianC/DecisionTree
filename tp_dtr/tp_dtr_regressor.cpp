//
//  tp_dtr_regressor.cpp
//  Classifer_RF
//
//  Created by jimmy on 2017-02-17.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "tp_dtr_regressor.h"
#include "tp_dtr_tree_node.h"

bool TPDTRegressor::predict(const Eigen::MatrixXf & feature,
                            Eigen::VectorXf & pred) const
{
    assert(trees_.size() > 0);
    assert(feature_dim_ == feature.cols());
    assert(feature_channel_ == feature.rows());
    
    pred = Eigen::VectorXf::Zero(label_dim_);
    
    // average predictions from all trees
    int pred_num = 0;
    for (int i = 0; i<trees_.size(); i++) {
        Eigen::VectorXf cur_pred;
        bool is_pred = trees_[i]->predict(feature, cur_pred);
        if (is_pred) {
            pred += cur_pred;
            pred_num++;
        }
    }
    if (pred_num == 0) {
        return false;
    }
    pred /= pred_num;
    return true;
}

bool TPDTRegressor::save(const char *fileName) const
{
    assert(trees_.size() > 0);
    // write tree number and tree files to file Name
    FILE *pf = fopen(fileName, "w");
    if(!pf) {
        printf("Error: can not open file %s\n", fileName);
        return false;
    }
    fprintf(pf, "%d %d %d\n", feature_channel_, feature_dim_, label_dim_);
    reg_tree_param_.writeToFile(pf);
    vector<string> tree_files;
    string baseName = string(fileName);
    baseName = baseName.substr(0, baseName.size()-4);
    for (int i = 0; i<trees_.size(); i++) {
        char buf[1024] = {NULL};
        sprintf(buf, "_%08d", i);
        string fileName = baseName + string(buf) + string(".txt");
        fprintf(pf, "%s\n", fileName.c_str());
        tree_files.push_back(fileName);
    }
    
    for (int i = 0; i<trees_.size(); i++) {
        if (trees_[i]) {
            TPDTRTreeNode::writeTree(tree_files[i].c_str(), trees_[i]->root_);
        }
    }
    fclose(pf);
    printf("save to %s\n", fileName);
    return true;
}

bool TPDTRegressor::load(const char *fileName)
{
    FILE *pf = fopen(fileName, "r");
    if (!pf) {
        printf("Error: can not open file %s\n", fileName);
        return false;
    }
    
    int ret_num = fscanf(pf, "%d %d %d", &feature_channel_, &feature_dim_, &label_dim_);
    assert(ret_num == 3);
    
    bool is_read = reg_tree_param_.readFromFile(pf);
    assert(is_read);
    reg_tree_param_.printSelf();
    
    vector<string> treeFiles;
    for (int i = 0; i<reg_tree_param_.tree_num_; i++) {
        char buf[1024] = {NULL};
        fscanf(pf, "%s", buf);
        treeFiles.push_back(string(buf));
    }
    fclose(pf);
    
    for (int i = 0; i<trees_.size(); i++) {
        delete trees_[i];
        trees_[i] = 0;
    }
    trees_.clear();
    
    // read each tree
    for (int i = 0; i<treeFiles.size(); i++) {
        Node * root = NULL;
        bool isRead = false;
        isRead = Node::readTree(treeFiles[i].c_str(), root);
        assert(isRead);
        assert(root);
        
        TreePtr tree = new Tree();
        tree->root_ = root;
        tree->setTreeParameter(reg_tree_param_);
        trees_.push_back(tree);
    }
    printf("read from %s\n", fileName);

    return true;    
}