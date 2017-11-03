//
//  dt_regressor.cpp
//  SequentialRandomForest
//
//  Created by jimmy on 2017-11-01.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "dt_regressor.h"
#include "dt_util.hpp"

DTRegressor::DTRegressor()
{
    
}

DTRegressor::~DTRegressor()
{
    for (int i = 0; i<trees_.size(); i++) {
        if (trees_[i]) {
            delete trees_[i];
            trees_[i] = NULL;
        }
    }
}

bool DTRegressor::predict(const Eigen::VectorXf & feature,
                           Eigen::VectorXf & pred) const
{
    assert(trees_.size() > 0);
    assert(feature.size() == tree_param_.feature_dimension_);
    
    vector<Eigen::VectorXf> predictions;
    for (int i = 0; i<trees_.size(); i++) {
        Eigen::VectorXf cur_pred;
        trees_[i]->predict(feature, cur_pred);
        predictions.push_back(cur_pred);
    }
    pred = predictions[0];
    for (int i = 1; i<predictions.size(); i++) {
        pred += predictions[i];
    }
    pred /= predictions.size();
    return true;
}

bool DTRegressor::save(const char *fileName) const
{
    assert(trees_.size() > 0);
    // write tree number and tree files to file Name
    FILE *pf = fopen(fileName, "w");
    if(!pf) {
        printf("Error: can not open file %s\n", fileName);
        return false;
    }
    
    tree_param_.writeToFile(pf);
    vector<string> tree_files;
    string baseName = string(fileName);
    baseName = baseName.substr(0, baseName.size()-4);
    for (int i = 0; i<trees_.size(); i++) {
        char buf[1024] = {NULL};
        sprintf(buf, "_%06d", i);
        string fileName = baseName + string(buf) + string(".txt");
        fprintf(pf, "%s\n", fileName.c_str());
        tree_files.push_back(fileName);
    }
    
    for (int i = 0; i<trees_.size(); i++) {
        if (trees_[i]) {
            trees_[i]->writeTree(tree_files[i].c_str());
        }
    }
    fclose(pf);
    printf("save to %s\n", fileName);
    return true;
}

bool DTRegressor::load(const char *fileName)
{
    FILE *pf = fopen(fileName, "r");
    if (!pf) {
        printf("Error: can not open file %s\n", fileName);
        return false;
    }
    
    bool is_read = tree_param_.readFromFile(pf);
    assert(is_read);
    // tree_param_.printSelf();
    
    vector<string> treeFiles;
    for (int i = 0; i<tree_param_.tree_num_; i++) {
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
        TreePtr pTree = new TreeType();
        assert(pTree);
        bool is_read = pTree->readTree(treeFiles[i].c_str());
        assert(is_read);
        pTree->setTreeParameter(tree_param_);
        trees_.push_back(pTree);
        
    }
    printf("read from %s\n", fileName);
    return true;
}