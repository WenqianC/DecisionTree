//
//  bt_rnd_tree.cpp
//  RGBD_RF
//
//  Created by jimmy on 2017-01-18.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "bt_rnd_tree.h"
#include "bt_rnd_tree_node.h"
#include "DTRandom.h"
#include "cvx_util.hpp"
#include "BTDTRUtil.h"
#include <iostream>


using std::cout;
using std::endl;

BTRNDTree::BTRNDTree()
{
    root_ = NULL;
    leaf_node_num_ = 0;    
}

BTRNDTree::~BTRNDTree()
{
    if (root_) {
        delete root_;
        root_ = NULL;
    }
}

bool BTRNDTree::buildTree(const vector<FeatureType> & features,
                          const vector<VectorXf> & labels,
                          const vector<unsigned int> & indices,
                          const vector<cv::Mat> & rgb_images,
                          const BTRNDTreeParameter & param)
{
    
    assert(indices.size() <= features.size());
    assert(labels.size() == features.size());
    
    root_ = new BTRNDTreeNode(0);
    tree_param_ = param;
    leaf_node_num_ = 0;
    this->configureNode(features, labels, rgb_images, indices, root_);
    assert(leaf_node_num_ > 0);
    
    // record leaf node
    this->hashLeafNode();
    return true;
}

bool BTRNDTree::configureNode(const vector<FeatureType> & features,
                              const vector<VectorXf> & labels,
                              const vector<cv::Mat> & rgb_images,
                              const vector<unsigned int> & indices,                              
                              NodePtr node)
{
    assert(indices.size() <= features.size());
    int depth = node->depth_;
    
    if (depth >= tree_param_.max_depth_ || indices.size() <= tree_param_.min_leaf_node_) {
        return this->setLeafNode(features, labels, indices, node);
    }
    
    // check standard deviation, early stop
    if (depth > tree_param_.max_depth_/2) {
        
        double variance = BTDTRUtil::spatial_variance(labels, indices);
        double std_dev = sqrt(variance/indices.size());
        if (std_dev < tree_param_.min_split_node_std_dev_) {
            return this->setLeafNode(features, labels, indices, node);
        }
    }
    
    // split samples into left and right node using random feature
    vector<unsigned int> left_indices;
    vector<unsigned int> right_indices;
    RandomSplitParameter rnd_split_param;
    double min_loss = this->optimizeRandomFeature(features, labels, rgb_images, indices, tree_param_,
                                                  depth,
                                                  left_indices, right_indices, rnd_split_param);
    
    bool is_split = min_loss < std::numeric_limits<double>::max();
    if (is_split) {
        assert(left_indices.size() + right_indices.size() == indices.size());
        if (tree_param_.verbose_ && depth < tree_param_.max_balanced_depth_) {
            printf("left, right node number is %lu %lu, percentage: %f \n", left_indices.size(), right_indices.size(),
                   100.0*left_indices.size()/indices.size());
        }
        // store split parameters
        node->split_param_ = rnd_split_param;
        if (left_indices.size() != 0) {
            NodePtr left_node = new Node(depth + 1);
            this->configureNode(features, labels, rgb_images, left_indices, left_node);
            node->left_child_ = left_node;
        }
        if (right_indices.size() != 0) {
            Node *right_node = new Node(depth + 1);   // increase depth
            this->configureNode(features, labels, rgb_images, right_indices,  right_node);
            node->right_child_ = right_node;
        }
        return true;
    }
    else
    {
        return this->setLeafNode(features, labels, indices, node);
    }
    
    return true;
}

double BTRNDTree::optimizeRandomFeature(const vector<FeatureType> & features,
                                        const vector<VectorXf> & labels,
                                        const vector<cv::Mat> & rgb_images,
                                        const vector<unsigned int> & indices,
                                        const BTRNDTreeParameter & tree_param,
                                        const int depth,
                                        vector<unsigned int> & left_indices,   //output
                                        vector<unsigned int> & right_indices,
                                        RandomSplitParameter & split_param)
{
    // split samples into left and right node
    const int max_pixel_offset = tree_param.max_pixel_offset_;
    const int max_channel = 3;
    const int max_random_num   = tree_param.pixel_offset_candidate_num_;
    
    double min_loss = std::numeric_limits<double>::max();
    for (int i = 0; i<max_random_num; i++) {
        double x2 = rnd_generator_.getRandomNumber(-max_pixel_offset, max_pixel_offset);
        double y2 = rnd_generator_.getRandomNumber(-max_pixel_offset, max_pixel_offset);
        
        RandomSplitParameter cur_split_param;
        cur_split_param.offset_[0] = x2;
        cur_split_param.offset_[1] = y2;
        cur_split_param.split_channles_[0] = rand()%max_channel;
        cur_split_param.split_channles_[1] = rand()%max_channel;
        
        
        vector<unsigned int> cur_left_indices;
        vector<unsigned int> cur_right_indices;
        double cur_loss = this->bestSplitRandomParameter(features, labels, rgb_images, indices, tree_param,
                                                         depth,
                                                         cur_split_param, cur_left_indices, cur_right_indices);
        
        if (cur_loss < min_loss) {
            min_loss = cur_loss;
            left_indices  = cur_left_indices;
            right_indices = cur_right_indices;
            split_param = cur_split_param;
        }
    }
    return min_loss;
}

double BTRNDTree::computeRandomFeature(const cv::Mat & rgb_image, const FeatureType * feat, const RandomSplitParameter & split)
{
    Eigen::Vector2f p1 = feat->p2d_;
    Eigen::Vector2f p2 = feat->addOffset(split.offset_);
    
    const int c1 = split.split_channles_[0];
    const int c2 = split.split_channles_[1];
    
    int p1x = p1[0];
    int p1y = p1[1];
    int p2x = p2[0];
    int p2y = p2[1];
    
    bool is_inside_image2 = CvxUtil::isInside(rgb_image.cols, rgb_image.rows, p2x, p2y);
    double pixel_1_c = 0.0;   // out of image as black pixels, random pixel values
    double pixel_2_c = 0.0;
    
    // pixel value at (y, x) [c] y is vertical, x is horizontal
    pixel_1_c = (rgb_image.at<cv::Vec3b>(p1y, p1x))[c1]; // (row, col)
    
    if (is_inside_image2) {
        pixel_2_c = (rgb_image.at<cv::Vec3b>(p2y, p2x))[c2];
    }
    return pixel_1_c - pixel_2_c;
}

double
BTRNDTree::bestSplitRandomParameter(const vector<FeatureType> & features,
                                    const vector<VectorXf> & labels,
                                    const vector<cv::Mat> & rgb_images,
                                    const vector<unsigned int> & indices,
                                    const BTRNDTreeParameter & tree_param,
                                    const int depth,
                                    RandomSplitParameter & split_param,
                                    vector<unsigned int> & left_indices,
                                    vector<unsigned int> & right_indices)
{
    double min_loss = std::numeric_limits<double>::max();
    
    const int min_node_size = tree_param.min_leaf_node_;
    const int split_candidate_num = tree_param.split_candidate_num_;
    const int max_balance_depth = tree_param.max_balanced_depth_;
    
    // calculate pixel difference
    vector<double> feature_values(indices.size(), 0.0); // 0.0 for invalid pixels
    for (int i = 0; i<indices.size(); i++) {
        int index = indices[i];
        assert(index >= 0 && index < features.size());
        
        const FeatureType* smp = &(features[index]);  // avoid copy, use pointer
        feature_values[i] = BTRNDTree::computeRandomFeature(rgb_images[smp->image_index_], smp, split_param);
        
    }
    
    double min_v = *std::min_element(feature_values.begin(), feature_values.end());
    double max_v = *std::max_element(feature_values.begin(), feature_values.end());
    if (!(min_v < max_v)) {
        return min_loss;
    }
    
    //printf("debug: min max values: %lf %lf\n", min_v, max_v);
    vector<double> split_values = rnd_generator_.getRandomNumbers(min_v, max_v, split_candidate_num);  // num_split_random = 20
    // printf("number of randomly selected spliting values is %lu\n", split_values.size());
    
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
        
        if (depth <= max_balance_depth) {
            cur_loss = BTDTRUtil::inbalance_loss((int)cur_left_index.size(), (int)cur_right_index.size());
        }
        else {
            cur_loss = BTDTRUtil::spatial_variance(labels, cur_left_index);
            if (cur_loss > min_loss) {
                continue;
            }
            cur_loss += BTDTRUtil::spatial_variance(labels, cur_right_index);
        }
        
        if (cur_loss < min_loss) {
            is_split = true;
            min_loss = cur_loss;
            left_indices  = cur_left_index;
            right_indices = cur_right_index;
            split_param.threshold_ = split_v;
        }
    }
    if (!is_split) {
        return min_loss;
    }
    assert(left_indices.size() + right_indices.size() == indices.size());
    
    return min_loss;
}


bool BTRNDTree::setLeafNode(const vector<FeatureType> & features,
                            const vector<VectorXf> & labels,
                            const vector<unsigned int> & indices,
                            NodePtr node)
{
    // descriptor of local patch, different from random feature
    vector<Eigen::VectorXf> local_features(indices.size());
    for (int i = 0; i<indices.size(); i++) {
        int idx = indices[i];
        local_features[i] = features[idx].x_descriptor_;
    }
    
    node->is_leaf_ = true;
    BTDTRUtil::mean_stddev<Eigen::VectorXf>(labels, indices, node->label_mean_, node->label_stddev_);
    node->feat_mean_ = BTDTRUtil::mean<Eigen::VectorXf>(local_features);
    leaf_node_num_++;
    
    
    if (tree_param_.verbose_leaf_) {
        printf("leaf node depth size %d    %lu\n", node->depth_, indices.size());
        cout<<"mean  : \n"<<node->label_mean_.transpose()<<endl;
        cout<<"stddev: \n"<<node->label_stddev_.transpose()<<endl;
    }
    return true;
}


void BTRNDTree::hashLeafNode()
{
    assert(leaf_node_num_ > 0);
    leaf_nodes_.resize(leaf_node_num_);
    
    int index = 0;
    this->recordLeafNodes(root_, leaf_nodes_, index);
    //printf("tree leaf node number is %d\n", leaf_node_num_);
}

void BTRNDTree::recordLeafNodes(NodePtr node, vector<NodePtr> & leafNodes, int & index)
{
    assert(node);
    if (node->is_leaf_) {
        // for tree read from a file, index is precomputed
        if (node->index_ != -1) {
            assert(node->index_ == index);
        }
        node->index_ = index;
        leafNodes[index] = node;
        index++;
        return;
    }
    if (node->left_child_) {
        this->recordLeafNodes(node->left_child_, leafNodes, index);
    }
    if (node->right_child_) {
        this->recordLeafNodes(node->right_child_, leafNodes, index);
    }
}

bool BTRNDTree::predict(const FeatureType & feature,
                        const cv::Mat & rgb_image,
                        const int maxCheck,
                        VectorXf & pred,
                        float & dist) const
{
    assert(root_);
    
    int checkCount = 0;
    const int knn = 1;
    
    BranchSt branch;
    flann::Heap<BranchSt> * heap = new flann::Heap<BranchSt>(leaf_node_num_*2);  // why use so large heap
    flann::DynamicBitset checked(leaf_node_num_);
    
    flann::KNNResultSet2<DistanceType> result(knn); // only keep the nearest one
    const ElementType *vec = feature.x_descriptor_.data();
    
    // search tree down to leaf
    this->searchLevel(result, vec, root_, checkCount, maxCheck, heap, checked, feature, rgb_image);
    
    while (heap->popMin(branch) &&
           (checkCount < maxCheck || !result.full())) {
        assert(branch.node);
        this->searchLevel(result, vec, branch.node, checkCount, maxCheck, heap, checked, feature, rgb_image);
    }
    
    delete heap;
    assert(result.size() == knn);
    
    size_t index = 0;
    DistanceType distance;
    result.copy(&index, &distance, 1, false);
    
    pred = leaf_nodes_[index]->label_mean_;
    dist = (float)distance;
    return true;
}

void BTRNDTree::searchLevel(flann::ResultSet<DistanceType>  & result_set, const ElementType* vec, const NodePtr node,
                            int & check_count, const int max_check,
                            flann::Heap<BranchSt>* heap, flann::DynamicBitset& checked,
                            const FeatureType & feature,     // new added parameter
                            const cv::Mat & rgb_image) const
{
    // check leaf node
    if (node->is_leaf_) {
        int index = node->index_;  // store leaf node index
        if (checked.test(index) ||
            (check_count >= max_check && result_set.full())) {
            return;
        }
        checked.set(index);
        check_count++;
        
        // squared distance
        DistanceType dist = distance_(node->feat_mean_.data(), vec, node->feat_mean_.size());
        result_set.addPoint(dist, index);
        return;
    }
    
    // create a branch record for the branch not taken 
    // compare random feature. Exploiting uncertainty in regression forests for accurate camera relocalization
    double rnd_feat = BTRNDTree::computeRandomFeature(rgb_image, &feature, node->split_param_);
    DistanceType diff = rnd_feat - node->split_param_.threshold_;
    NodePtr bestChild  = (diff < 0 ) ? node->left_child_: node->right_child_;
    NodePtr otherChild = (diff < 0 ) ? node->right_child_: node->left_child_;
    
    // insert all possible branches, because the distance measurement in random feature and local patch feature are different
    DistanceType dist = (DistanceType)fabs(diff);
    if (!result_set.full()) {
        heap->insert(BranchSt(otherChild, dist));
    }
    
    // call recursively to search next level
    this->searchLevel(result_set, vec, bestChild, check_count, max_check, heap, checked, feature, rgb_image);
}

void BTRNDTree::getLeafNodeDescriptor(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & data)
{
    assert(root_);
    assert(leaf_node_num_ > 0);
    assert(leaf_node_num_ == leaf_nodes_.size());
    
    const int rows = leaf_node_num_;
    const int cols = (int)leaf_nodes_[0]->feat_mean_.size();
    
    data = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Zero(rows, cols);
    for (int i = 0; i<leaf_nodes_.size(); i++) {
        data.row(i) = leaf_nodes_[i]->feat_mean_;
    }
}

void BTRNDTree::setLeafNodeDescriptor(const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & data)
{
    assert(root_);
    assert(leaf_node_num_ > 0);
    assert(leaf_node_num_ == data.rows());
    
    this->hashLeafNode();
    for (int i = 0; i<leaf_nodes_.size(); i++) {
        leaf_nodes_[i]->feat_mean_ = data.row(i);
    }
}



