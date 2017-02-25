//
//  tp_dtr_tree.cpp
//  Classifer_RF
//
//  Created by jimmy on 2017-02-16.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "tp_dtr_tree.h"
#include "tp_dtr_tree_node.h"
#include <iostream>
#include "dt_util.h"


using std::cout;
using std::endl;

TPDTRTree::~TPDTRTree()
{
    if (root_) {
        delete root_;
        root_ = NULL;
    }
    
}


bool TPDTRTree::buildTree(const vector<MatrixXf> & features,
                          const vector<VectorXf> & labels,
                          const vector<unsigned int> & indices,
                          const TreeParameter & param)
{
    assert(features.size() == labels.size());
    assert(indices.size() <= features.size());
    
    tree_param_ = param;
    root_ = new Node(0);
    int channels = (int)features.front().rows();
    assert(channels == 3);
    
    // generate permutation
    vector<int> pairwise_cmp = {-1, 0, 1};
    trinary_permutation_ = TPDTRUtil::generatePermutation(pairwise_cmp);
    trinary_permutation_.resize(trinary_permutation_.size()/2);
    vector<int> uniary = {0, 0, 1};
    vector<vector<int> > unary = TPDTRUtil::generatePermutation(uniary);
    trinary_permutation_.insert(trinary_permutation_.end(), unary.begin(), unary.end());
    
    cout<<"feature projection permutation begin: "<<endl;
    for(int i = 0; i<trinary_permutation_.size(); i++) {
        for (auto c: trinary_permutation_[i]) {
            cout<<c<<" ";
        }
        cout<<endl;
    }
    cout<<"feature projection permutation end. "<<endl;
    
    return this->configureNode(features, labels, indices, root_);
}

bool
TPDTRTree::bestSplitParameter(const vector<Eigen::MatrixXf> & features,
                                    const vector<VectorXf> & labels,
                                    const vector<unsigned int> & indices,
                                    const TPDTRTreeParameter & tree_param,
                                    const int depth,
                                    TPDTRSplitParameter & split_param,
                                    vector<unsigned int> & left_indices,
                                    vector<unsigned int> & right_indices)
{
    double min_loss = std::numeric_limits<double>::max();
    
    const int min_node_size = tree_param.min_leaf_node_;
    const int candidate_threshold_num = tree_param.candidate_threshold_num_;
    const int proj_dim = (int)features.front().rows();
    const int max_balanced_depth = tree_param.max_balanced_depth_;
    assert(proj_dim == split_param.split_weight_.size());
    
    vector<int> wt = split_param.split_weight_;
    int split_dim  = split_param.split_dim_;
    
    // calculate projected feature values
    vector<double> feature_values(indices.size(), 0.0); // 0.0 for invalid pixels
    for (int i = 0; i<indices.size(); i++) {
        int index = indices[i];
        assert(index >= 0 && index < features.size());
        double val = 0.0;
        for (int j = 0; j<proj_dim; j++) {
            val += wt[j] * features[index](j, split_dim);
        }
        feature_values[i] = val;
    }
    
    double min_v = *std::min_element(feature_values.begin(), feature_values.end());
    double max_v = *std::max_element(feature_values.begin(), feature_values.end());
    if (!(min_v < max_v)) {
        return false;
    }
    
    // random split values
    vector<double> split_values = rnd_generator_.getRandomNumbers(min_v, max_v, candidate_threshold_num);
     
    // split data by pixel difference
    bool is_split = false;
    for (int i = 0; i<split_values.size(); i++) {
        double split_v = split_values[i];
        vector<unsigned int> cur_left_index;
        vector<unsigned int> cur_right_index;
        double cur_loss = 0;
        for (int j = 0; j<feature_values.size(); j++) {
            int index = indices[j];
            if (feature_values[j] < split_v) {
                cur_left_index.push_back(index);
            }
            else
            {
                cur_right_index.push_back(index);
            }
        }
        assert(cur_left_index.size() + cur_right_index.size() == indices.size());
        
        // avoid too small internal node
        if (cur_left_index.size() < min_node_size/2 || cur_right_index.size() < min_node_size/2) {
            continue;
        }
        
        if (depth <= max_balanced_depth) {
            cur_loss = DTUtil::balanceLoss((int)cur_left_index.size(), (int)cur_right_index.size());
        }
        else {
            cur_loss = DTUtil::spatialVariance<VectorXf>(labels, cur_left_index);
            if (cur_loss > min_loss) {
                continue;
            }
            cur_loss += DTUtil::spatialVariance<VectorXf>(labels, cur_right_index);
        }
        
        if (cur_loss < min_loss) {
            is_split = true;
            min_loss = cur_loss;
            left_indices  = cur_left_index;
            right_indices = cur_right_index;
            split_param.split_threshold_ = split_v;
            split_param.split_loss_ = min_loss;
            assert(left_indices.size() + right_indices.size() == indices.size());
        }
    }
    
    if (is_split) {
        assert(left_indices.size() + right_indices.size() == indices.size());
    }
    
    return is_split;
}


bool TPDTRTree::configureNode(const vector<MatrixXf> & features,
                              const vector<VectorXf> & labels,
                              const vector<unsigned int> & indices,
                              NodePtr node)
{
    assert(indices.size() <= features.size());
    assert(node);
    
    const int min_leaf_node = tree_param_.min_leaf_node_;
    const int max_depth     = tree_param_.max_tree_depth_;
    const int depth = node->depth_;
    const int dim = (int)features[0].cols();
    const int candidate_dim_num = tree_param_.candidate_dim_num_;
    const double min_split_stddev = tree_param_.min_split_node_std_dev_;
    const int candidate_projection_num = tree_param_.candidate_projection_num_;
    
    assert(candidate_dim_num <= dim);
    assert(candidate_projection_num <= trinary_permutation_.size());
    
    
    if (depth >= max_depth || indices.size() <= min_leaf_node) {
        return this->setLeafNode(features, labels, indices, node);
    }
    
    // check standard deviation, early stop
    if (depth > max_depth/2) {
        double variance = DTUtil::spatialVariance<VectorXf>(labels, indices);
        double std_dev = sqrt(variance/indices.size());
        if (std_dev < min_split_stddev) {
            return this->setLeafNode(features, labels, indices, node);
        }
    }
    
    // randomly select a subset of dimensions
    vector<unsigned int> dims;
    for (unsigned int i = 0; i<dim; i++) {
        dims.push_back(i);
    }
    std::random_shuffle(dims.begin(), dims.end());
    vector<unsigned int> random_dim(dims.begin(), dims.begin() + candidate_dim_num);
    assert(random_dim.size() > 0 && random_dim.size() <= dims.size());
    
    // randomly select a subset of trinary projection
    vector<unsigned int> projection_index;
    for (unsigned int i = 0; i<trinary_permutation_.size(); i++) {
        projection_index.push_back(i);
    }
    std::random_shuffle(projection_index.begin(), projection_index.end());
    projection_index.resize(candidate_projection_num);
    vector<vector<int> > random_projection;          // random projection 
    for (int i = 0; i<projection_index.size(); i++) {
        random_projection.push_back(trinary_permutation_[projection_index[i]]);
    }
    
    
    // split the data to left and right node
    vector<unsigned int> left_indices;
    vector<unsigned int> right_indices;
    TPDTRSplitParameter split_param;
    bool is_split = false;
    double loss = std::numeric_limits<double>::max();
    
    // optimize random feature
    for (int i = 0; i<random_dim.size(); i++) {
        TPDTRSplitParameter cur_split_param;
        cur_split_param.split_dim_ = random_dim[i];
        for (int j = 0; j<random_projection.size(); j++) {
            cur_split_param.split_weight_ = random_projection[j];
            vector<unsigned int> cur_left_indices;
            vector<unsigned int> cur_right_indices;
           
            // split data once
            bool cur_split = this->bestSplitParameter(features, labels, indices, tree_param_, depth, cur_split_param,
                                                       cur_left_indices, cur_right_indices);
            if (cur_split && (cur_split_param.split_loss_ < loss)) {
                is_split = true;
                loss = cur_split_param.split_loss_;
                split_param = cur_split_param;
                left_indices = cur_left_indices;
                right_indices = cur_right_indices;
                assert(left_indices.size() + right_indices.size() == indices.size());
            }
        }
    }
    
    if (is_split) {
        assert(left_indices.size() + right_indices.size() == indices.size());
        if (tree_param_.verbose_) {
            printf("left, right node number is %lu %lu, percentage: %f \n", left_indices.size(), right_indices.size(),
                   100.0*left_indices.size()/indices.size());
        }
        // store split parameters
        node->split_param_ = split_param;
        if (left_indices.size() != 0) {
            NodePtr left_node = new Node(depth + 1);
            this->configureNode(features, labels, left_indices, left_node);
            node->left_child_ = left_node;
        }
        if (right_indices.size() != 0) {
            Node *right_node = new Node(depth + 1);   // increase depth
            this->configureNode(features, labels, right_indices,  right_node);
            node->right_child_ = right_node;
        }
    }
    else {
        return this->setLeafNode(features, labels, indices, node);
    }
    
    return true;
}



bool TPDTRTree::setLeafNode(const vector<Eigen::MatrixXf> & features,
                            const vector<VectorXf> & labels,
                            const vector<unsigned int> & indices,
                            NodePtr node)
{
    node->is_leaf_ = true;
    DTUtil::meanStddev<Eigen::VectorXf>(labels, indices, node->label_mean_, node->label_stddev_);
    node->split_param_.split_weight_.resize(features.front().rows(), 0);    
    if (tree_param_.verbose_leaf_) {
        printf("leaf node depth size %d    %lu\n", node->depth_, indices.size());
        cout<<"mean  : \n"<<node->label_mean_.transpose()<<endl;
        cout<<"stddev: \n"<<node->label_stddev_.transpose()<<endl;
    }
    return true;
}


bool TPDTRTree::predict(const Eigen::MatrixXf & feature,
                        Eigen::VectorXf & pred) const
{
    assert(root_);
    return this->predict(root_, feature, pred);
}

bool TPDTRTree::predict(const NodePtr node,
                        const Eigen::MatrixXf & feature,
                        Eigen::VectorXf & pred) const
{
    if (node->is_leaf_) {
        pred = node->label_mean_;
        return true;
    }
    
    double feat = 0.0;
    
    int dim = node->split_param_.split_dim_;
    assert(dim < feature.cols());
    
    vector<int> wt = node->split_param_.split_weight_;
    assert(wt.size() == feature.rows());
    // projection feature
    for (int i = 0; i<wt.size(); i++) {
        feat += wt[i] * feature(i, dim);
    }
    
    NodePtr best_node  = (feat < node->split_param_.split_threshold_) ? node->left_child_: node->right_child_;
    NodePtr other_node = (feat < node->split_param_.split_threshold_) ? node->right_child_: node->left_child_;
    
    if (best_node) {
        return this->predict(best_node, feature, pred);
    }
    else if (other_node) {
        //return this->predict(other_node, feature, pred);
        return false;
    }
    else {
        return false;
    }
}

const TPDTRTree::TreeParameter & TPDTRTree::getTreeParameter(void) const
{
    return tree_param_;
}

void TPDTRTree::setTreeParameter(const TreeParameter & param)
{
    tree_param_ = param;
}