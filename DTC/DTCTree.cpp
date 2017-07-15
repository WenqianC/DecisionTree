//
//  DTCTree.cpp
//  Classifer_RF
//
//  Created by jimmy on 2016-07-26.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#include "DTCTree.h"
#include "DTCNode.h"
#include <algorithm>
#include <iostream>

using std::cout;
using std::endl;

bool DTCTree::buildTree(const vector<VectorXd> & features,
                        const vector<unsigned int> & labels,
                        const vector<unsigned int> & indices,
                        const DTCTreeParameter & param)
{
    assert(features.size() == labels.size());
    assert(indices.size() <= features.size());
    
    tree_param_ = param;
    root_ = new DTCNode(0);
    
    return this->configureNode(features, labels, indices, root_);
}

static bool bestSplitDimesion(const vector<VectorXd> & features,
                              const vector<unsigned int> & labels,
                              const vector<unsigned int> & indices,
                              const DTCTreeParameter & tree_param,
                              DTCSplitParameter & split_param,
                              vector<unsigned int> & left_indices,
                              vector<unsigned int> & right_indices)
{
    // randomly select number in a range
    const int dim = split_param.split_dim_;
    double min_v = std::numeric_limits<double>::max();
    double max_v = std::numeric_limits<double>::min();
    const int rand_num = tree_param.split_candidate_num_;
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
    vector<double> rnd_split_values = DTCUtil::generateRandomNumber(min_v, max_v, rand_num);
    
    bool is_split = false;
    double loss = std::numeric_limits<double>::max();
    const int min_split_num = tree_param.min_split_num_;
    const int category_num  = tree_param.category_num_;
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
        
        // at least one example
        // probability of each category (label) in left node
        Eigen::VectorXd left_prob = Eigen::VectorXd::Ones(category_num);
        for (int j = 0; j<cur_left_indices.size(); j++) {
            int label = labels[cur_left_indices[j]];
            left_prob[label] += 1.0;
        }
        left_prob /= cur_left_indices.size() + category_num;
        
        Eigen::VectorXd right_prob = Eigen::VectorXd::Ones(category_num);
        for (int j = 0; j<cur_right_indices.size(); j++) {
            int label = labels[cur_right_indices[j]];
            right_prob[label] += 1.0;
        }
        right_prob /= cur_right_indices.size() + category_num;
        
        double left_entropy  = DTCUtil::crossEntropy(left_prob);
        double right_entropy = DTCUtil::crossEntropy(right_prob);
        double left_ratio = 1.0 * cur_left_indices.size()/indices.size();
        double entropy = left_ratio * left_entropy + (1.0 - left_ratio) * right_entropy;        
        
        if (entropy < loss) {
            loss = entropy;
            is_split = true;
            left_indices = cur_left_indices;
            right_indices = cur_right_indices;
            split_param.split_threshold_ = threshold;
            split_param.split_loss_ = entropy;
        }        
    }
    
    return is_split;
}


bool DTCTree::configureNode(const vector<VectorXd> & features,
                            const vector<unsigned int> & labels,
                            const vector<unsigned int> & indices,                           
                            DTCNode * node)
{
    assert(node);
    const int min_lead_node = tree_param_.min_leaf_node_num_;
    const int max_depth     = tree_param_.max_depth_;
    const int category_num  = tree_param_.category_num_;
    const int depth = node->depth_;
    const int dim = (int)features[0].size();
    
    // leaf node
    if (indices.size() < min_lead_node || depth > max_depth ||
        ( depth > max_depth/2 && DTCUtil::isSameLabel(labels, indices))) {
        node->is_leaf_ = true;
        Eigen::VectorXd prob = Eigen::VectorXd::Zero(category_num);
        for (int i = 0; i<indices.size(); i++) {
            assert(labels[indices[i]] >= 0 && labels[indices[i]] < category_num);
            prob[labels[indices[i]]] += 1.0;
        }
        prob /= indices.size();
        node->prob_ = prob;
        if (tree_param_.verbose_leaf_) {
            printf("leaf node depth size %d    %lu\n", node->depth_, indices.size());
            cout<<"probability: \n"<<node->prob_.transpose()<<endl<<endl;;
        }
        node->sample_num_ = (int)indices.size();
        return true;
    }
     
    
    // randomly select a subset of dimensions
    vector<unsigned int> dims;
    for (unsigned int i = 0; i<dim; i++) {
        dims.push_back(i);
    }
    int sqrt_feat_dim = sqrt((double)dim);
    std::random_shuffle(dims.begin(), dims.end());
    vector<unsigned int> random_dim(dims.begin(), dims.begin() + sqrt_feat_dim);
    assert(random_dim.size() > 0 && random_dim.size() <= dims.size());
    
    // split the data to left and right node
    vector<unsigned int> left_indices;
    vector<unsigned int> right_indices;
    DTCSplitParameter split_param;
    bool is_split = false;
    double loss = std::numeric_limits<double>::max();
    
    // optimize random feature
    for (int i = 0; i<random_dim.size(); i++) {
        DTCSplitParameter cur_split_param;
        cur_split_param.split_dim_ = random_dim[i];
        
        vector<unsigned int> cur_left_indices;
        vector<unsigned int> cur_right_indices;
        
        bool cur_is_split = bestSplitDimesion(features, labels, indices, tree_param_, cur_split_param, cur_left_indices, cur_right_indices);
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
        node->is_leaf_ = false;
        node->sample_num_ = (int)indices.size();
        if (left_indices.size() > 0) {
            DTCNode *left_node = new DTCNode(depth + 1);
            this->configureNode(features, labels, left_indices, left_node);
            left_node->sample_percentage_ = 1.0 * left_indices.size()/indices.size();
            node->left_child_ = left_node;
        }
        if (right_indices.size() > 0) {
            DTCNode * right_node = new DTCNode(depth + 1);
            this->configureNode(features, labels, right_indices, right_node);
            right_node->sample_percentage_ = 1.0 * right_indices.size()/indices.size();
            node->right_child_ = right_node;
        }
        
    }
    else
    {
        node->is_leaf_ = true;
        Eigen::VectorXd prob = Eigen::VectorXd::Zero(category_num);
        for (int i = 0; i<indices.size(); i++) {
            assert(labels[indices[i]] >= 0 && labels[indices[i]] < category_num);
            prob[labels[indices[i]]] += 1.0;
        }
        prob /= indices.size();
        node->prob_ = prob;
        if (tree_param_.verbose_leaf_) {
            printf("leaf node depth size %d    %lu\n", node->depth_, indices.size());
            cout<<"probability: \n"<<node->prob_.transpose()<<endl<<endl;;
        }
        node->sample_num_ = (int)indices.size();
    }    
    return true;
}

bool DTCTree::predict(const Eigen::VectorXd & feature,
                      Eigen::VectorXd & prob) const
{
    assert(feature.size() == tree_param_.feature_dimension_);
    assert(root_);
    
    return this->predict(root_, feature, prob);
}

bool DTCTree::predict(const Eigen::VectorXd & feature,
                      unsigned int & pred) const
{
    Eigen::VectorXd prob;
    bool isPred = this->predict(feature, prob);
    if (!isPred) {
        return false;
    }
    prob.maxCoeff(&pred);
    return true;    
}

void DTCTree::computeProximity(const vector<Eigen::VectorXd> & features,
                               const vector<unsigned int> & indices,
                               DTProximity & proximity) const
{
    assert(root_);
    this->computeProximity(root_, features, indices, proximity);    
}

const DTCTreeParameter & DTCTree::getTreeParameter(void) const
{
    return tree_param_;
}

void DTCTree::setTreeParameter(const DTCTreeParameter & param)
{
    tree_param_ = param;
}


bool DTCTree::predict(const DTCNode * node,
                      const Eigen::VectorXd & feature,
                      Eigen::VectorXd & prob) const
{
    assert(node);
    if (node->is_leaf_) {
        prob = node->prob_;
        return true;
    }
    double feat = feature[node->split_param_.split_dim_];
    if (feat < node->split_param_.split_threshold_ && node->left_child_) {
        return this->predict(node->left_child_, feature, prob);
    }
    else if (node->right_child_)
    {
        return this->predict(node->right_child_, feature, prob);
    }
    else
    {
        printf("Warning: prediction can not find proper split value\n");
        return false;
    }
}

void DTCTree::computeProximity(const DTCNode * node,
                               const vector<Eigen::VectorXd> & features,
                               const vector<unsigned int> & indices,
                               DTProximity & proximity) const
{
    assert(node);
    if (node->is_leaf_) {
        for (int i = 0; i<indices.size(); i++) {
            for (int j = i+1; j<indices.size(); j++) {
                proximity.addExample(indices[i], indices[j]);
            }
        }
        return;
    }
    
    int dim = node->split_param_.split_dim_;
    double threshold = node->split_param_.split_threshold_;
    vector<unsigned int> left_indices;
    vector<unsigned int> right_indices;
    for (int i = 0; i<indices.size(); i++) {
        double feat = features[indices[i]][dim];
        if (feat < threshold) {
            left_indices.push_back(indices[i]);
        }
        else {
            right_indices.push_back(indices[i]);
        }
    }
    assert(left_indices.size() + right_indices.size() == indices.size());
    
    if (node->left_child_ && left_indices.size() > 0) {
        this->computeProximity(node->left_child_, features, left_indices, proximity);
    }
    if (node->right_child_ && right_indices.size() > 0) {
        this->computeProximity(node->right_child_, features, right_indices, proximity);
    }
}


