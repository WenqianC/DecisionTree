//
//  tp_dt_classifier.cpp
//  Classifer_RF
//
//  Created by jimmy on 2017-02-19.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "tp_dt_classifier.h"
#include "tp_dtc_tree_node.h"

bool TPDTClassifier::predict(const Eigen::MatrixXf & feature,
                          Eigen::VectorXf & prob) const
{
    assert(trees_.size() > 0);
    assert(feature.rows() == feature_channel_);
    assert(feature.cols() == feature_dim_);
    
    prob = Eigen::VectorXf::Zero(category_num_);
    
    int pred_num = 0;
    for (int i = 0; i<trees_.size(); i++) {
        Eigen::VectorXf p;
        bool is_pred = trees_[i]->predict(feature, p);
        if (is_pred) {
            prob += p;
            pred_num++;
        }
    }
    prob /= pred_num;
    
    return (pred_num != 0);
}

bool TPDTClassifier::predict(const Eigen::MatrixXf & feature,
                            unsigned int & pred) const
{
    assert(trees_.size() > 0);
    assert(feature_dim_ == feature.cols());
    assert(feature_channel_ == feature.rows());
    
    Eigen::VectorXf prob;
    bool isPred = this->predict(feature, prob);
    if (!isPred) {
        return false;
    }
    prob.maxCoeff(&pred);
    return true;
}

bool TPDTClassifier::save(const char *fileName) const
{
    assert(trees_.size() > 0);
    // write tree number and tree files to file Name
    FILE *pf = fopen(fileName, "w");
    if(!pf) {
        printf("Error: can not open file %s\n", fileName);
        return false;
    }
    fprintf(pf, "%d %d %d\n", feature_channel_, feature_dim_, category_num_);
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
            TPDTCTreeNode::writeTree(tree_files[i].c_str(), trees_[i]->root_);
        }
    }
    fclose(pf);
    printf("save to %s\n", fileName);
    return true;
}

bool TPDTClassifier::load(const char *fileName)
{
    FILE *pf = fopen(fileName, "r");
    if (!pf) {
        printf("Error: can not open file %s\n", fileName);
        return false;
    }
    
    int ret_num = fscanf(pf, "%d %d %d", &feature_channel_, &feature_dim_, &category_num_);
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

