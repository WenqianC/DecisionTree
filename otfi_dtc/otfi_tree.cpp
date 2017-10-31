//
//  otfi_tree.cpp
//  SequentialRandomForest
//
//  Created by jimmy on 2017-07-16.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "otfi_tree.hpp"
#include <iostream>
#include "dt_util.hpp"

using std::cout;
using std::endl;

OTFITree::OTFITree()
{
    root_ = NULL;
    feature_dims_ = 0;
}

OTFITree::~OTFITree()
{
    if (root_) {
        delete root_;
        root_ = NULL;
    }
}

const OTFITree::TreeParameter & OTFITree::getTreeParameter(void) const
{
    return tree_param_;
}

void OTFITree::setTreeParameter(const TreeParameter & param)
{
    tree_param_ = param;
}

bool OTFITree::buildTree(const vector<Eigen::VectorXf> & features,
                         const vector<int> & labels,
                         const vector<int> & indices,
                         const TreeParameter & param)
{
    assert(features.size() == labels.size());
    assert(indices.size() <= features.size());
   
    
    tree_param_ = param;
    feature_dims_ = (int)features[0].size();
    root_ = new Node(0);
    
    return this->buildTreeImpl(features, labels, indices, root_);
}

bool OTFITree::buildTreeImpl(const vector<Eigen::VectorXf> & features,
                             const vector<int> & labels,
                             const vector<int> & indices,
                             NodePtr node)
{
    // leaf node
    assert(indices.size() <= features.size());
    assert(node);
    
    const int min_leaf_node = tree_param_.min_leaf_node_num_;
    const int max_depth     = tree_param_.max_tree_depth_;
    const int candidate_dim_num = tree_param_.candidate_dim_num_;
    const int depth = node->depth_;
    const int dim = (int)features[0].size();
    
    // Step 1: check if stop data splitting, [0, max_depth)
    if (depth >= max_depth || indices.size() <= min_leaf_node) {
        return this->setLeafNode(features, labels, indices, node);
    }
    
    // randomly select a subset of dimensions
    vector<int> dims = DTUtil::range<int>(0, dim, 1);
    std::random_shuffle(dims.begin(), dims.end());
    vector<unsigned int> random_dim(dims.begin(), dims.begin() + candidate_dim_num);
    assert(random_dim.size() > 0 && random_dim.size() <= dims.size());
    
    // split the data to left and right node
    vector<int> left_indices;
    vector<int> right_indices;
    SplitParameter split_param;
    bool is_split = false;
    double loss = std::numeric_limits<double>::max();
    
    // optimize random feature
    for (int i = 0; i<random_dim.size(); i++) {
        SplitParameter cur_split_param;
        cur_split_param.dim_ = random_dim[i];
        
        vector<int> cur_left_indices;
        vector<int> cur_right_indices;
        // split data once
        bool cur_split = this->bestSplitParameter(features, labels, indices, cur_split_param,
                                                  cur_left_indices, cur_right_indices);
        
        if (cur_split && (cur_split_param.loss_ < loss)) {
            is_split = true;
            loss = cur_split_param.loss_;
            split_param = cur_split_param;
            left_indices = cur_left_indices;
            right_indices = cur_right_indices;
            assert(left_indices.size() + right_indices.size() == indices.size());
        }
    }
    
    if (is_split) {
        assert(left_indices.size() + right_indices.size() == indices.size());
        if (tree_param_.verbose_) {
            printf("left, right node number is %lu %lu, percentage: %f \n",
                   left_indices.size(),
                   right_indices.size(),
                   100.0*left_indices.size()/indices.size());
        }
        node->split_param_ = split_param;
        
        if (left_indices.size() != 0) {
            NodePtr left_node = new Node(depth + 1);
            this->buildTreeImpl(features, labels, left_indices, left_node);
            node->left_child_ = left_node;
        }
        if (right_indices.size() != 0) {
            Node *right_node = new Node(depth + 1);   // increase depth
            this->buildTreeImpl(features, labels, right_indices,  right_node);
            node->right_child_ = right_node;
        }
    }
    else {
        return this->setLeafNode(features, labels, indices, node);
    }
    
    return true;
}

bool OTFITree::bestSplitParameter(const vector<VectorXf> & features,
                              const vector<int> & labels,
                              const vector<int> & indices,
                              SplitParameter & split_param,
                              vector<int> & left_indices,
                              vector<int> & right_indices)
{
    
    const int dim = split_param.dim_;
    double min_v = std::numeric_limits<double>::max();
    double max_v = std::numeric_limits<double>::min();
    const int rand_num = tree_param_.candidate_threshold_num_;
    const int min_split_num = tree_param_.min_leaf_node_num_/2 + 1;
    const int category_num  = tree_param_.category_num_;
    
    // randomly select number in a range
    for (int i = 0; i<indices.size(); i++) {
        double v = features[indices[i]][dim];
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
    vector<double> rnd_split_values = rnd_generator_.generateRandomNumber(min_v, max_v, rand_num);
    split_param.lower_bound_ = min_v;
    split_param.upper_bound_ = max_v;
    
    // optimize the threshold
    bool is_split = false;
    double loss = std::numeric_limits<double>::max();
    for (int i = 0; i<rnd_split_values.size(); i++) {
        double threshold = rnd_split_values[i];
        vector<int> cur_left_indices;
        vector<int> cur_right_indices;
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
        
        if (cur_left_indices.size() < min_split_num ||
            cur_right_indices.size() < min_split_num) {
            continue;
        }
        
        // probability of each category (label) in left node
        Eigen::VectorXd left_prob = Eigen::VectorXd::Zero(category_num);
        for (int j = 0; j<cur_left_indices.size(); j++) {
            int label = labels[cur_left_indices[j]];
            left_prob[label] += 1.0;
        }
        left_prob /= cur_left_indices.size();
        
        Eigen::VectorXd right_prob = Eigen::VectorXd::Zero(category_num);
        for (int j = 0; j<cur_right_indices.size(); j++) {
            int label = labels[cur_right_indices[j]];
            right_prob[label] += 1.0;
        }
        right_prob /= cur_right_indices.size();
        
        double left_entropy  = DTUtil::crossEntropy(left_prob);
        double right_entropy = DTUtil::crossEntropy(right_prob);
        double left_ratio = 1.0 * cur_left_indices.size()/indices.size();
        double entropy = left_ratio * left_entropy + (1.0 - left_ratio) * right_entropy;
        
        if (entropy < loss) {
            loss = entropy;
            is_split = true;
            left_indices = cur_left_indices;
            right_indices = cur_right_indices;
            split_param.threshold_ = threshold;
            split_param.loss_ = entropy;
        }
    }
    return is_split;
}

bool OTFITree::setLeafNode(const vector<Eigen::VectorXf> & features,
                             const vector<int> & labels,
                             const vector<int> & indices,
                             NodePtr node)
{    
    const int category_num = tree_param_.category_num_;    
    node->is_leaf_ = true;
    
    Eigen::VectorXf prob = Eigen::VectorXf::Zero(category_num);
    for (int i = 0; i<indices.size(); i++) {
        const int label = labels[indices[i]];   // tree depth is related to time step
        assert(label >= 0 && label < category_num);
        prob[label] += 1.0;
    }
    prob /= indices.size();
    node->prob_ = prob;
    
    if (tree_param_.verbose_leaf_) {
        printf("leaf node depth size %d    %lu\n", node->depth_, indices.size());
        cout<<"probability: "<<node->prob_.transpose()<<endl<<endl;;
    }
    return true;
}

bool OTFITree::imputeFeature(const vector<Eigen::VectorXf> & features,
                             const vector<int> & labels,
                             const vector<int> & indices,
                             
                             const vector<Eigen::VectorXf> & mdata_features,
                             const vector<int> & mdata_labels,
                             const vector<int> & mdata_indices,
                             const float mdata_mask,
                             vector<Eigen::VectorXf> & imputed_features, // output
                             vector<float>& weight) const
{
    assert(root_);
    assert(mdata_features.size() == mdata_labels.size());
    assert(imputed_features.size() == mdata_features.size());
    assert(imputed_features.size() == weight.size());
    
    return this->imputeFeatureImpl(root_, features, labels, indices, 
                                   mdata_features, mdata_labels, mdata_indices, mdata_mask,
                                   imputed_features, weight);
}



static bool isSameValue(const float v1, const float v2)
{
    return fabsf(v1 - v2) < 0.00001;
}

bool OTFITree::imputeFeatureImpl(const NodePtr node,
                                 const vector<Eigen::VectorXf> & features,
                                 const vector<int> & labels,
                                 const vector<int> & indices,
                                 
                                 const vector<Eigen::VectorXf> & mdata_features,
                                 const vector<int> & mdata_labels,
                                 const vector<int> & mdata_indices,
                                 
                                 const float mdata_mask,
                                 vector<Eigen::VectorXf> & imputed_features,
                                 vector<float>& weight) const
{
    assert(node);
    if (node->is_leaf_) {
        assert(indices.size() > 0);
        // mean value of features
        Eigen::VectorXf feat_mean = DTUtil::mean<Eigen::VectorXf>(features, indices);
        
        // using a weighted version,
        for (int i = 0; i<mdata_indices.size(); i++) {
            int index = mdata_indices[i];
            int label = mdata_labels[index];
            assert(label >= 0 && label < node->prob_.size());
            assert(mdata_features[index].size() == feat_mean.size());
            
            // loop each dimension
            for (int d = 0; d < feat_mean.size(); d++) {
                // the missing value has same value as mdata_mask
                if (isSameValue(mdata_features[index][d],  mdata_mask)) {
                    imputed_features[index][d] = feat_mean[d];
                    weight[index] = node->prob_[label];                    
                }
            }
        }
        return true;
    }
    
    // split data by tree structure
    int dim = node->split_param_.dim_;
    double threshold = node->split_param_.threshold_;
    double lower_bound = node->split_param_.lower_bound_;
    double upper_bound = node->split_param_.upper_bound_;
    vector<int> left_indices;
    vector<int> right_indices;
    for (int i = 0; i<indices.size(); i++) {
        int index = indices[i];
        double feat = features[index][dim];
        if (feat < threshold) {
            left_indices.push_back(index);
        }
        else {
            right_indices.push_back(index);
        }
    }
    assert(left_indices.size() + right_indices.size() == indices.size());
    assert(left_indices.size() != 0 && right_indices.size() != 0);    
    
    vector<int> mdata_left_indices;
    vector<int> mdata_right_indices;
    for (int i = 0; i<mdata_indices.size(); i++) {
        int index = mdata_indices[i];
        float feat = mdata_features[index][dim];
        if (isSameValue(feat, mdata_mask)) {
            // randomly generate a value as feature
            feat = rnd_generator_.getRandomNumber(lower_bound, upper_bound);
        }
        if (feat < threshold) {
            mdata_left_indices.push_back(index);
        }
        else {
            mdata_right_indices.push_back(index);
        }
    }
    assert(mdata_left_indices.size() + mdata_right_indices.size() == mdata_indices.size());
    
    if (mdata_left_indices.size() > 0) {
        this->imputeFeatureImpl(node->left_child_,
                                features, labels, left_indices,
                                mdata_features, mdata_labels, mdata_left_indices,
                                mdata_mask,
                                imputed_features, weight);
    }
    
    if (mdata_right_indices.size() > 0) {
        this->imputeFeatureImpl(node->right_child_,
                                features, labels, right_indices,
                                mdata_features, mdata_labels, mdata_right_indices,
                                mdata_mask,
                                imputed_features, weight);
    }
    
    return true;
}

bool OTFITree::predict(const Eigen::VectorXf & feature,
                       Eigen::VectorXf & prob) const
{
    assert(root_);
    return this->predictImpl(root_, feature, prob);
}

bool OTFITree::predict(const Eigen::VectorXf & feature,
                       int & pred) const
{
    Eigen::VectorXf prob;
    bool isPred = this->predict(feature, prob);
    if (!isPred) {
        return false;
    }
    prob.maxCoeff(&pred);
    return true;
}

bool OTFITree::predictImpl(const NodePtr node,
                           const Eigen::VectorXf & feature,
                           Eigen::VectorXf & prob) const
{
    assert(node);
    if (node->is_leaf_) {
        prob = node->prob_;
        return true;
    }
    double feat = feature[node->split_param_.dim_];
    if (feat < node->split_param_.threshold_ && node->left_child_) {
        return this->predictImpl(node->left_child_, feature, prob);
    }
    else if (node->right_child_) {
        return this->predictImpl(node->right_child_, feature, prob);
    }
    else {
        printf("Warning: prediction can not find proper split value\n");
        return false;
    }
}

