//
//  bt_rnd_regressor.cpp
//  RGBD_RF
//
//  Created by jimmy on 2017-01-21.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "bt_rnd_regressor.h"
#include "bt_rnd_tree.h"
#include "yael_io.h"
#include "bt_rnd_tree_node.h"
#include "cvxUtil.hpp"


bool BTRNDRegressor::predict(const Feature & feature,
                             const cv::Mat & rgb_image,
                             const int max_check,
                             vector<Eigen::VectorXf> & predictions,
                             vector<float> & dists) const
{
    assert(trees_.size() > 0);
    assert(predictions.size() == 0);
    assert(dists.size() == 0);
    
    vector<Eigen::VectorXf> unordered_predictions;
    vector<float> unordered_dists;
    for (int i = 0; i<trees_.size(); i++) {
        Eigen::VectorXf cur_pred;
        float dist;
        bool is_pred = trees_[i]->predict(feature, rgb_image, max_check, cur_pred, dist);
        if (is_pred) {
            unordered_predictions.push_back(cur_pred);
            unordered_dists.push_back(dist);
        }
    }
    
    // order by color distance
    vector<size_t> sortIndexes = CvxUtil::sortIndices<float>(unordered_dists);
    for (int i = 0; i<sortIndexes.size(); i++) {
        predictions.push_back(unordered_predictions[sortIndexes[i]]);
        dists.push_back(unordered_dists[sortIndexes[i]]);
    }
    
    return predictions.size() > 0;
    assert(predictions.size() == dists.size());    
    return predictions.size() == trees_.size();
}

const BTRNDTreeParameter & BTRNDRegressor::getTreeParameter(void) const
{
    return reg_tree_param_;
}

const DatasetParameter & BTRNDRegressor::getDatasetParameter(void) const
{
    return dataset_param_;
}
const BTRNDTree * BTRNDRegressor::getTree(int index) const
{
    assert(index >=0 && index < trees_.size());
    return trees_[index];
}

bool BTRNDRegressor::save(const char *fileName) const
{
    assert(trees_.size() > 0);
    // write tree number and tree files to file Name
    FILE *pf = fopen(fileName, "w");
    if(!pf) {
        printf("Error: can not open file %s\n", fileName);
        return false;
    }
    fprintf(pf, "%d\n", label_dim_);
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
    
    // leaf node feature
    vector<string> leaf_node_files;
    for (int i = 0; i<trees_.size(); i++) {
        char buf[1024] = {NULL};
        sprintf(buf, "_%08d", i);
        string fileName = baseName + string(buf) + string(".fvec");
        fprintf(pf, "%s\n", fileName.c_str());
        leaf_node_files.push_back(fileName);
    }
    for (int i = 0; i<trees_.size(); i++) {
        if (trees_[i]) {
            Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> data;
            // get descriptors from leaf node
            trees_[i]->getLeafNodeDescriptor(data);
            YaelIO::write_fvecs_file(leaf_node_files[i].c_str(), data);
        }
    }
    
    for (int i = 0; i<trees_.size(); i++) {
        if (trees_[i]) {
            BTRNDTreeNode::writeTree(tree_files[i].c_str(), trees_[i]->root_, trees_[i]->leaf_node_num_);
        }
    }
    fclose(pf);
    printf("save to %s\n", fileName);
    return true;
}

bool BTRNDRegressor::load(const char *fileName)
{
    FILE *pf = fopen(fileName, "r");
    if (!pf) {
        printf("Error: can not open file %s\n", fileName);
        return false;
    }
    
    int ret_num = fscanf(pf, "%d", &label_dim_);
    assert(ret_num == 1);
    
    bool is_read = reg_tree_param_.readFromFile(pf);
    assert(is_read);
    reg_tree_param_.printSelf();
    
    // read tree file
    vector<string> treeFiles;
    for (int i = 0; i<reg_tree_param_.tree_num_; i++) {
        char buf[1024] = {NULL};
        fscanf(pf, "%s", buf);
        treeFiles.push_back(string(buf));
    }
    
    // read leaf node descriptor file
    vector<string> leaf_node_files;
    for (int i = 0; i<reg_tree_param_.tree_num_; i++) {
        char buf[1024] = {NULL};
        fscanf(pf, "%s", buf);
        leaf_node_files.push_back(string(buf));
    }
    fclose(pf);
    
    for (int i = 0; i<trees_.size(); i++) {
        if (trees_[i]) {
            delete trees_[i];
            trees_[i] = NULL;
        }
    }
    trees_.clear();
    
    // read each tree
    for (int i = 0; i<treeFiles.size(); i++) {
        BTRNDTreeNode * root = NULL;
        int leaf_node_num = 0;
        bool isRead = false;
        isRead = BTRNDTreeNode::readTree(treeFiles[i].c_str(), root, leaf_node_num);
        assert(isRead);
        assert(root);
        
        BTRNDTree *tree = new BTRNDTree();
        tree->root_ = root;
        tree->setTreeParameter(reg_tree_param_);
        tree->leaf_node_num_ = leaf_node_num;
        
        // read leaf node descriptor and set it in the tree
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> data;
        isRead = YaelIO::read_fvecs_file(leaf_node_files[i].c_str(), data);
        assert(isRead);
        tree->setLeafNodeDescriptor(data);
        
        trees_.push_back(tree);
    }
    printf("read from %s\n", fileName);
    return true;
}
