//
//  DTRTree.cpp
//  Classifer_RF
//
//  Created by jimmy on 2016-10-03.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#include "DTRTree.h"
#include "DTRNode.h"
#include "DTRUtil.h"
#include "DTCUtil.h"
#include <iostream>

using std::cout;
using std::endl;

bool DTRTree::buildTree(const vector<VectorXd> & features,
                        const vector<VectorXd> & labels,
                        const vector<unsigned int> & indices,
                        const DTRTreeParameter & param)
{
    assert(features.size() == labels.size());
    assert(indices.size() <= features.size());
    
    tree_param_ = param;
    root_ = new DTRNode(0);
    
    return this->configureNode(features, labels, indices, root_);
}

static bool bestSplitDimension(const vector<VectorXd> & features,
                               const vector<VectorXd> & labels,
                               const vector<unsigned int> & indices,
                               const DTRTreeParameter & tree_param,
                               DTRSplitParameter & split_param,
                               vector<unsigned int> & left_indices,
                               vector<unsigned int> & right_indices)
{
    // randomly select number in a range
    const int dim = split_param.split_dim_;
    double min_v = std::numeric_limits<double>::max();
    double max_v = std::numeric_limits<double>::min();
    const int threshold_num = tree_param.candidate_threshold_num_;
    for (int i = 0; i<indices.size(); i++) {
        int index = indices[i];
        double v = features[index][dim];
        if (v > max_v) {
            max_v = v;
        }
        if (v < min_v) {
            min_v = v;
        }
    }
    if (!(min_v < max_v)) {
        return false;
    }
    vector<double> rnd_split_values = DTCUtil::generateRandomNumber(min_v, max_v, threshold_num);
    
    bool is_split = false;
    double loss = std::numeric_limits<double>::max();
    const int min_split_num = tree_param.min_split_node_;
    for (int i = 0; i<rnd_split_values.size(); i++) {
        double threshold = rnd_split_values[i];
        vector<unsigned int> cur_left_indices;
        vector<unsigned int> cur_right_indices;
        // split data by comparing with the threshold
        for (int j = 0; j<indices.size(); j++) {
            int index = indices[j];
            double v = features[index][dim];
            if (v < threshold) {
                cur_left_indices.push_back(index);
            }
            else {
                cur_right_indices.push_back(index);
            }
        }
        
        if (cur_left_indices.size() < min_split_num || cur_right_indices.size() < min_split_num) {
            continue;
        }
     
        double cur_loss = 0.0;
        cur_loss += DTRUtil::spatial_variance(labels, cur_left_indices);
        cur_loss += DTRUtil::spatial_variance(labels, cur_right_indices);
        
        
        if (cur_loss < loss) {
            loss = cur_loss;
            is_split = true;
            left_indices = cur_left_indices;
            right_indices = cur_right_indices;
            split_param.split_threshold_ = threshold;
            split_param.split_loss_ = cur_loss;
        }
    }
    
    return is_split;
}


bool DTRTree::configureNode(const vector<VectorXd> & features,
                            const vector<VectorXd> & labels,
                            const vector<unsigned int> & indices,
                            DTRNode * node)
{
    assert(node);
    const int min_leaf_node = tree_param_.min_leaf_node_;
    const int max_depth     = tree_param_.max_tree_depth_;
    const int depth = node->depth_;
    const int dim = (int)features[0].size();
    const int candidate_dim_num = tree_param_.candidate_dim_num_;
    const double min_split_stddev = tree_param_.min_split_node_std_dev_;
    assert(candidate_dim_num <= dim);
    
    // leaf node
    bool reach_leaf = false;
    if (indices.size() < min_leaf_node || depth > max_depth) {
        reach_leaf = true;
    }
    
    // check standard deviation
    if (reach_leaf == false && depth > max_depth/2) {
        double variance = DTRUtil::spatial_variance(labels, indices);
        double std_dev = sqrt(variance/indices.size());
        if (std_dev < min_split_stddev) {
            reach_leaf = true;
        }
    }    
    // satisfy leaf node
    if (reach_leaf) {
        node->is_leaf_ = true;
        DTRUtil::mean_stddev(labels, indices, node->mean_, node->stddev_);
        if (tree_param_.verbose_leaf_) {
            printf("leaf node depth size %d    %lu\n", node->depth_, indices.size());
            cout<<"mean  : \n"<<node->mean_<<endl;
            cout<<"stddev: \n"<<node->stddev_<<endl;
        }
        node->sample_num_ = (int)indices.size();
        return true;
    }
    
    // randomly select a subset of dimensions
    vector<unsigned int> dims;
    for (unsigned int i = 0; i<dim; i++) {
        dims.push_back(i);
    }
    std::random_shuffle(dims.begin(), dims.end());
    vector<unsigned int> random_dim(dims.begin(), dims.begin() + candidate_dim_num);
    assert(random_dim.size() > 0 && random_dim.size() <= dims.size());
    
    // split the data to left and right node
    vector<unsigned int> left_indices;
    vector<unsigned int> right_indices;
    DTRSplitParameter split_param;
    bool is_split = false;
    double loss = std::numeric_limits<double>::max();
    
    
    // optimize random feature
    for (int i = 0; i<random_dim.size(); i++) {
        DTRSplitParameter cur_split_param;
        cur_split_param.split_dim_ = random_dim[i];
        
        vector<unsigned int> cur_left_indices;
        vector<unsigned int> cur_right_indices;
        
        bool cur_is_split = bestSplitDimension(features, labels, indices, tree_param_, cur_split_param, cur_left_indices, cur_right_indices);
        if (cur_is_split && cur_split_param.split_loss_ < loss) {
            is_split = true;
            loss = cur_split_param.split_loss_;
            split_param = cur_split_param;
            left_indices = cur_left_indices;
            right_indices = cur_right_indices;
        }
    }
    
    // split data
    if (is_split) {
        assert(left_indices.size() + right_indices.size() == indices.size());
        if (tree_param_.verbose_) {
            printf("percentage is %f\n", 1.0 * left_indices.size()/indices.size());
            printf("loss is %f \n", split_param.split_loss_);
        }
        node->split_param_ = split_param;
        node->sample_num_ = (int)indices.size();
        node->is_leaf_ = false;
        if (left_indices.size() > 0) {
            DTRNode *left_node = new DTRNode(depth + 1);
            this->configureNode(features, labels, left_indices, left_node);
            left_node->sample_percentage_ = 1.0 * left_indices.size()/indices.size();
            node->left_child_ = left_node;
        }
        if (right_indices.size() > 0) {
            DTRNode * right_node = new DTRNode(depth + 1);
            this->configureNode(features, labels, right_indices, right_node);
            right_node->sample_percentage_ = 1.0 * right_indices.size()/indices.size();
            node->right_child_ = right_node;
        }
    }
    else
    {
        node->is_leaf_ = true;
        DTRUtil::mean_stddev(labels, indices, node->mean_, node->stddev_);
        if (tree_param_.verbose_leaf_) {
            printf("leaf node depth size %d    %lu\n", node->depth_, indices.size());
            cout<<"mean  : \n"<<node->mean_<<endl;
            cout<<"stddev: \n"<<node->stddev_<<endl;
        }
        node->sample_num_ = (int)indices.size();
        return true;
    }
    
    return true;
}

const DTRTreeParameter & DTRTree::getTreeParameter(void) const
{
    return tree_param_;
}

void DTRTree::setTreeParameter(const DTRTreeParameter & param)
{
    tree_param_ = param;
}


bool DTRTree::predict(const Eigen::VectorXd & feature,
                      Eigen::VectorXd & pred) const
{
    assert(root_);
    return this->predict(root_, feature, pred);
}

bool DTRTree::predict(const DTRNode * node,
                      const Eigen::VectorXd & feature,
                      Eigen::VectorXd & pred) const
{
    assert(node);
    if (node->is_leaf_) {
        pred = node->mean_;
        return true;
    }
    assert(node->split_param_.split_dim_ < feature.size());
    double feat = feature[node->split_param_.split_dim_];
    if (feat < node->split_param_.split_threshold_ && node->left_child_) {
        return this->predict(node->left_child_, feature, pred);
    }
    else if (node->right_child_)
    {
        return this->predict(node->right_child_, feature, pred);
    }
    else
    {
        printf("Warning: prediction can not find proper split value\n");
        return false;
    }

    return true;
}



