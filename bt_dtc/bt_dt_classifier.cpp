//
//  bt_dt_classifier.cpp
//  PTZBTRF
//
//  Created by jimmy on 2017-10-13.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "bt_dt_classifier.h"
#include "yael_io.h"

BTDTClassifier::BTDTClassifier()
{
    feature_dim_ = 0;
    category_num_ = 0;
}

BTDTClassifier::~BTDTClassifier()
{
    for(int i = 0; i<trees_.size(); i++)
    {
        if (trees_[i]) {
            delete trees_[i];
            trees_[i] = NULL;
        }
    }
}
bool BTDTClassifier::predict(const Eigen::VectorXf & feature,
                             const int max_check,
                             vector<int> & predictions,
                             vector<float> & dists) const
             
{
    assert(trees_.size() > 0);
    assert(feature_dim_ == feature.size());
    assert(predictions.size() == 0);
    assert(dists.size() == 0);
    
    // predict from each tree
    for (int i = 0; i<trees_.size(); i++) {
        int cur_pred = 0;
        float dist = 0;
        bool is_pred = trees_[i]->predict(feature, max_check, cur_pred, dist);
        if (is_pred) {
            predictions.push_back(cur_pred);
            dists.push_back(dist);
        }
    }
    assert(predictions.size() == dists.size());
    return predictions.size() == trees_.size();
}

bool BTDTClassifier::predict(const Eigen::VectorXf & feature,
                             const int max_check,
                             int & prediction,
                             const float dist_threshold)
{
    vector<int> predictions;
    vector<float> dists;
    
    prediction = -1; // default as no valid prediction
    
    bool is_pred = this->predict(feature, max_check, predictions, dists);
    if (!is_pred) {
        return false;
    }
    
    Eigen::VectorXi label_distribtion = Eigen::VectorXi::Zero(category_num_, 1);
    int valid_num = 0;
    for (int i = 0; i<dists.size(); i++) {
        if (dists[i] < dist_threshold) {
            valid_num++;
            label_distribtion[predictions[i]]++;
        }
    }
    if (valid_num == 0) {
        return false;
    }
    
    // majority vote
    label_distribtion.maxCoeff(&prediction);
    return true;
}

bool BTDTClassifier::save(const char *file_name) const
{
    assert(trees_.size() > 0);
    
    // write tree files to file
    FILE *pf = fopen(file_name, "w");
    if(!pf) {
        printf("Error: can not open file %s\n", file_name);
        return false;
    }
    
    tree_param_.writeToFile(pf); // must write tree parameter first
    fprintf(pf, "%d %d\n", feature_dim_, category_num_);
    
    vector<string> tree_files;
    string base_name = string(file_name);
    base_name = base_name.substr(0, base_name.size()-4);
    for (int i = 0; i<trees_.size(); i++) {
        char buf[1024] = {NULL};
        sprintf(buf, "_%08d", i);
        string name = base_name + string(buf) + string(".txt");
        fprintf(pf, "%s\n", name.c_str());
        tree_files.push_back(name);
    }
    
    // leaf node feature
    vector<string> leaf_node_files;
    for (int i = 0; i<trees_.size(); i++) {
        char buf[1024] = {NULL};
        sprintf(buf, "_%08d", i);
        string name = base_name + string(buf) + string(".fvec");
        fprintf(pf, "%s\n", name.c_str());
        leaf_node_files.push_back(name);
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
            trees_[i]->writeTree(tree_files[i].c_str());
        }
    }
    fclose(pf);
    printf("save to %s\n", file_name);
    return true;
}

bool BTDTClassifier::load(const char *file_name)
{
    FILE *pf = fopen(file_name, "r");
    if (!pf) {
        printf("Error: can not open file %s\n", file_name);
        return false;
    }
    
    bool is_read = tree_param_.readFromFile(pf);
    assert(is_read);
    tree_param_.printSelf();
    
    int ret_num = fscanf(pf, "%d %d", &feature_dim_, &category_num_);
    assert(ret_num == 2);
    
    // read tree file
    vector<string> tree_files;
    for (int i = 0; i<tree_param_.tree_num_; i++) {
        char buf[1024] = {NULL};
        ret_num = fscanf(pf, "%s", buf);
        assert(ret_num == 1);
        tree_files.push_back(string(buf));
    }
    
    // read leaf node descriptor file
    vector<string> leaf_node_files;
    for (int i = 0; i<tree_param_.tree_num_; i++) {
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
    for (int i = 0; i<tree_files.size(); i++) {
        
        
        TreePtr pTree = new TreeType();
        pTree->readTree(tree_files[i].c_str());
        pTree->setTreeParameter(tree_param_);
        
        // read leaf node descriptor and set it in the tree
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> data;
        bool is_read = YaelIO::read_fvecs_file(leaf_node_files[i].c_str(), data);
        assert(is_read);
        pTree->setLeafNodeDescriptor(data);
        trees_.push_back(pTree);
    }
    printf("read from %s\n", file_name);
    return true;
}
