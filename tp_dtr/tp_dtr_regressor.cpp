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
    return true;    
}