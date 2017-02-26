//
//  seq_dtr_tree.cpp
//  Classifer_RF
//
//  Created by jimmy on 2017-02-25.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "seq_dtr_tree.h"
#include "seq_dtr_tree_node.h"
#include <iostream>
#include "dt_util.h"


using std::cout;
using std::endl;

SeqDTRTree::~SeqDTRTree()
{
    if (root_) {
        delete root_;
        root_ = NULL;
    }
}


bool SeqDTRTree::buildTree(const vector<MatrixXf> & feature_seqs,
                           const vector<MatrixXf > & label_seqs,
                           const vector<unsigned int> & indices,
                           const TreeParameter & param,
                           const vector<unsigned int>& time_steps)
{
    assert(feature_seqs.size() == label_seqs.size());
    assert(indices.size() <= feature_seqs.size());
    assert(label_seqs[0].size() == feature_seqs[0].rows());
    
    tree_param_ = param;
    time_steps_ = time_steps;
    weights_.resize(time_steps_.size(), 1.0);
    root_ = new Node(0);
    
    assert(time_steps_.size() == tree_param_.max_tree_depth_);
    
    return this->configureNode(feature_seqs, label_seqs, indices, root_);
}


bool
SeqDTRTree::bestSplitParameter(const vector<Eigen::MatrixXf> & feature_seqs,
                               const vector<Eigen::MatrixXf > & label_seqs,
                               const vector<unsigned int> & indices,
                               const SeqDTRTreeParameter & tree_param,
                               const int depth,
                               SplitParameter & split_param,
                               vector<unsigned int> & left_indices,
                               vector<unsigned int> & right_indices)
{
    
    double min_loss = std::numeric_limits<double>::max();
    const int min_node_size = tree_param.min_leaf_node_;
    const int candidate_threshold_num = tree_param.candidate_threshold_num_;
    const int time_step = split_param.split_time_step_;
    const int split_dim  = split_param.split_dim_;
    
    // calculate projected feature values
    vector<double> feature_values(indices.size(), 0.0); // 0.0 for invalid pixels
    for (int i = 0; i<indices.size(); i++) {
        int index = indices[i];
        assert(index >= 0 && index < feature_seqs.size());
        feature_values[i] = feature_seqs[index](time_step, split_dim);
    }
    
    double min_v = *std::min_element(feature_values.begin(), feature_values.end());
    double max_v = *std::max_element(feature_values.begin(), feature_values.end());
    if (!(min_v < max_v)) {
        return false;
    }
    
    // random split values
    vector<double> split_values = rnd_generator_.getRandomNumbers(min_v, max_v, candidate_threshold_num);
    
    bool is_split = false;
    for (int i = 0; i<split_values.size(); i++) {
        vector<unsigned int> cur_left_index;
        vector<unsigned int> cur_right_index;
        const double split_v = split_values[i];  // threshold
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
        
        // sum of variance in leaf and right nodes
        cur_loss += DTUtil::sumOfVariance<Eigen::MatrixXf>(label_seqs, time_step, cur_left_index);
        if (cur_loss > min_loss) {
            continue;
        }
        cur_loss += DTUtil::sumOfVariance<Eigen::MatrixXf>(label_seqs, time_step, cur_right_index);
        
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

bool SeqDTRTree::configureNode(const vector<MatrixXf> & feature_seqs,
                               const vector<MatrixXf > & label_seqs,
                               const vector<unsigned int> & indices,
                               NodePtr node)
{
    assert(indices.size() <= feature_seqs.size());
    assert(node);
    
    const int min_leaf_node = tree_param_.min_leaf_node_;
    const int max_depth     = tree_param_.max_tree_depth_;
    const int depth = node->depth_;
    const int dim = (int)feature_seqs[0].cols();
    const int candidate_dim_num = tree_param_.candidate_dim_num_;
    const int time_step = time_steps_[depth];       // sequential data
    
    assert(candidate_dim_num <= dim);
    
    // [0, max_depth)
    if (depth >= max_depth - 1|| indices.size() <= min_leaf_node) {
        return this->setLeafNode(label_seqs, indices, node);
    }      
    
    // randomly select a subset of dimensions
    vector<unsigned int> dims = DTUtil::range<unsigned int>(0, dim, 1);
    std::random_shuffle(dims.begin(), dims.end());
    vector<unsigned int> random_dim(dims.begin(), dims.begin() + candidate_dim_num);
    assert(random_dim.size() > 0 && random_dim.size() <= dims.size());
    
    
    // split the data to left and right node
    vector<unsigned int> left_indices;
    vector<unsigned int> right_indices;
    SeqDTRTree::SplitParameter split_param;
    bool is_split = false;
    double loss = std::numeric_limits<double>::max();
    
    // optimize random feature
    for (int i = 0; i<random_dim.size(); i++) {
        SeqDTRTree::SplitParameter cur_split_param;
        cur_split_param.split_dim_ = random_dim[i];
        cur_split_param.split_time_step_ = time_step;
        
        vector<unsigned int> cur_left_indices;
        vector<unsigned int> cur_right_indices;
        
        // split data and compute loss once
        bool cur_split = this->bestSplitParameter(feature_seqs, label_seqs, indices, tree_param_,
                                                  depth, cur_split_param,
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
    
    
    if (is_split) {
        assert(left_indices.size() + right_indices.size() == indices.size());
        if (tree_param_.verbose_) {
            printf("left, right node number is %lu %lu, percentage: %f \n",
                   left_indices.size(),
                   right_indices.size(),
                   100.0*left_indices.size()/indices.size());
        }
        // set internal node
        this->setInternalNode(label_seqs, indices, node, split_param);
        
        if (left_indices.size() != 0) {
            NodePtr left_node = new Node(depth + 1);
            this->configureNode(feature_seqs, label_seqs, left_indices, left_node);
            node->left_child_ = left_node;
        }
        if (right_indices.size() != 0) {
            Node *right_node = new Node(depth + 1);   // increase depth
            this->configureNode(feature_seqs, label_seqs, right_indices,  right_node);
            node->right_child_ = right_node;
        }
    }
    else {
        return this->setLeafNode(label_seqs, indices, node);
    }
    
    return true;
}

bool SeqDTRTree::setInternalNode(const vector<Eigen::MatrixXf> & label_seqs,
                                 const vector<unsigned int> & indices,
                                 NodePtr node,
                                 const SplitParameter & split_param)
{
    assert(node->depth_ < time_steps_.size());
    const int time_step = time_steps_[node->depth_];
    assert(time_step < label_seqs[0].rows());
    
    node->is_leaf_ = false;
    DTUtil::rowMeanStddev<Eigen::MatrixXf, Eigen::VectorXf>(label_seqs, indices, time_step, node->label_mean_, node->label_std_);
    node->split_param_ = split_param;
    
    return true;
}


bool SeqDTRTree::setLeafNode(const vector<Eigen::MatrixXf > & labels,
                             const vector<unsigned int> & indices,
                             NodePtr node)
{
    assert(node->depth_ < time_steps_.size());
    const int time_step = time_steps_[node->depth_];    
    assert(time_step < labels[0].rows());
    
    node->is_leaf_ = true;
    DTUtil::rowMeanStddev<Eigen::MatrixXf, Eigen::VectorXf>(labels, indices, time_step, node->label_mean_, node->label_std_);
    node->split_param_.split_time_step_ = time_step;
    
    if (tree_param_.verbose_leaf_) {
        printf("leaf node depth size %d    %lu\n", node->depth_, indices.size());
        cout<<"leaf mean: \n"<<node->label_mean_.transpose()<<"\n stddev: \n"<<node->label_std_.transpose()<<endl<<endl;
    }
    return true;
}

bool SeqDTRTree::rawPredict(const Eigen::MatrixXf & feature_seq,
                         vector<unsigned int> & time_steps,
                         vector<Eigen::VectorXf> & predictions) const
{
    assert(root_);
    return this->rawPredict(root_, feature_seq, time_steps, predictions);
}

bool SeqDTRTree::predict(const Eigen::MatrixXf & feature_seq,
                         vector<unsigned int> & time_steps,
                         vector<double> & weights,
                         vector<Eigen::VectorXf> & predictions) const
{
    assert(root_);
    assert(time_steps_.size() == weights_.size());    
    time_steps = time_steps_;
    weights = weights_;
    
    bool is_pred = this->predict(root_, feature_seq, predictions);
    if (!is_pred) {
        return false;
    }
    assert(predictions.size() == time_steps.size());    
    return true;
}

bool SeqDTRTree::rawPredict(const NodePtr node,
                            const Eigen::MatrixXf & feature_seq,
                            vector<unsigned int> & time_steps,
                            vector<Eigen::VectorXf> & predictions) const
{    
    assert(weights_.size() == 0 || weights_.size() == time_steps_.size());
    if (node->is_leaf_) {
        time_steps.push_back(node->split_param_.split_time_step_);
        predictions.push_back(node->label_mean_);
        
        // reach maximum depth
        for (int d = node->depth_ + 1; d < tree_param_.max_tree_depth_; d++) {
            time_steps.push_back(time_steps_[d]);
            predictions.push_back(node->label_mean_);
        }
        return true;
    }
    
    int step = node->split_param_.split_time_step_;  // time step
    int dim = node->split_param_.split_dim_;
    assert(step < feature_seq.rows());
    assert(dim < feature_seq.cols());
    double feat = feature_seq(step, dim);
    
    // output current (internal) node prediction
    time_steps.push_back(step);
    predictions.push_back(node->label_mean_);
    
    NodePtr best_node  = (feat < node->split_param_.split_threshold_) ? node->left_child_: node->right_child_;
    NodePtr other_node = (feat < node->split_param_.split_threshold_) ? node->right_child_: node->left_child_;
    
    if (best_node) {
        return this->rawPredict(best_node, feature_seq, time_steps, predictions);
    }
    else if (other_node) {
        return false;
    }
    else {
        return false;
    }
}

bool SeqDTRTree::predict(const NodePtr node,
                         const Eigen::MatrixXf & feature_seq,
                         vector<Eigen::VectorXf> & predictions) const
{
    assert(weights_.size() == time_steps_.size());
    
    // output current (internal) node prediction
    predictions.push_back(node->label_mean_);
    
    if (node->is_leaf_) {
        // reach maximum depth
        for (int d = node->depth_ + 1; d < tree_param_.max_tree_depth_; d++) {
            predictions.push_back(node->label_mean_);
        }
        return true;
    }
    
    int step = node->split_param_.split_time_step_;  // time step
    int dim = node->split_param_.split_dim_;
    assert(step < feature_seq.rows());
    assert(dim < feature_seq.cols());
    double feat = feature_seq(step, dim);
    
    NodePtr best_node  = (feat < node->split_param_.split_threshold_) ? node->left_child_: node->right_child_;
    // NodePtr other_node = (feat < node->split_param_.split_threshold_) ? node->right_child_: node->left_child_;
    
    if (best_node) {
        return this->predict(best_node, feature_seq, predictions);
    }    
    else {
        return false;
    }
}


bool SeqDTRTree::setWeights(const vector<double> & wts)
{
    weights_ = wts;
    return true;
}

const SeqDTRTree::TreeParameter & SeqDTRTree::getTreeParameter(void) const
{
    return tree_param_;
}

void SeqDTRTree::setTreeParameter(const TreeParameter & param)
{
    tree_param_ = param;
}

void SeqDTRTree::setTimesteps(const vector<unsigned int>& steps)
{
    time_steps_ = steps;
}