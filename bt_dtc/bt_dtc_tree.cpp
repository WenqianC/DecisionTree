//
//  bt_dtc_tree.cpp
//  PTZBTRF
//
//  Created by jimmy on 2017-10-13.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "bt_dtc_tree.h"
#include "dt_util.hpp"
#include <iostream>

using std::cout;
using std::endl;

BTDTCTree::BTDTCTree()
{
    root_ = NULL;
}

BTDTCTree::~BTDTCTree()
{
    if (root_) {
        delete root_;
    }
    
    root_ = NULL;
}

void BTDTCTree::getLeafNodeDescriptor(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & data)
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

void BTDTCTree::setLeafNodeDescriptor(const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & data)
{
    assert(root_);
    assert(leaf_node_num_ > 0);
    assert(leaf_node_num_ == data.rows());
    
    this->hashLeafNode();
    for (int i = 0; i<leaf_nodes_.size(); i++) {
        leaf_nodes_[i]->feat_mean_ = data.row(i);
    }
}

const BTDTCTree::TreeParameter & BTDTCTree::getTreeParameter(void) const
{
    return tree_param_;
}

void BTDTCTree::setTreeParameter(const TreeParameter & param)
{
    tree_param_ = param;
}


bool BTDTCTree::buildTree(const vector<VectorXf> & features,
                          const vector<int> & labels,
                          const vector<int> & indices,
                          const TreeParameter & param)
{
    assert(features.size() == labels.size());
    assert(indices.size() <= features.size());
    
    tree_param_ = param;
    root_ = new Node(0);
    leaf_node_num_ = 0;
    
    this->buildTreeImpl(features, labels, indices, root_);
    this->hashLeafNode();
    return true;
}

bool BTDTCTree::buildTreeImpl(const vector<VectorXf> & features,
                              const vector<int> & labels,
                              const vector<int> & indices,
                              NodePtr node)
{
    assert(node);
    const int min_leaf_size = tree_param_.min_leaf_size_;
    const int max_depth     = tree_param_.max_tree_depth_;
    const int depth = node->depth_;
    const int dim = (int)features[0].size();
    const int candidate_dim_num = tree_param_.candidate_dim_num_;
    assert(candidate_dim_num <= dim);
    
    // 1. check if reaches leaf node
    if (indices.size() < min_leaf_size || depth > max_depth) {
        this->setLeafNode(features, labels, indices, node);
        return true;
    }
    
    // 2. optimize splitting
    // randomly select a subset of dimensions
    vector<int> random_dim = dt::randomDimension(dim, candidate_dim_num);
    assert(random_dim.size() > 0 && random_dim.size() <= dim);
    
    // split the data to left and right node
    vector<int> left_indices;
    vector<int> right_indices;
    SplitParameter split_param;
    bool is_split = false;
    double loss = std::numeric_limits<double>::max();
    
    // optimize random feature
    for (int i = 0; i<random_dim.size(); i++) {
        SplitParameter cur_split_param;
        cur_split_param.split_dim_ = random_dim[i];
        
        vector<int> cur_left_indices;
        vector<int> cur_right_indices;
        bool cur_is_split = this->bestSplitParameter(features, labels, indices, depth,
                                                     cur_split_param,
                                                     cur_left_indices,
                                                     cur_right_indices);
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
            printf("left percentage is %f \n", 1.0 * left_indices.size()/indices.size());
            printf("split      loss is %f \n", split_param.split_loss_);
        }
        node->split_param_ = split_param;
        node->sample_num_ = (int)indices.size();
        node->is_leaf_ = false;
        if (left_indices.size() > 0) {
            NodePtr left_node = new Node(depth + 1);
            this->buildTreeImpl(features, labels, left_indices, left_node);
            left_node->sample_percentage_ = 1.0 * left_indices.size()/indices.size();
            node->left_child_ = left_node;
        }
        if (right_indices.size() > 0) {
            NodePtr right_node = new Node(depth + 1);
            this->buildTreeImpl(features, labels, right_indices, right_node);
            right_node->sample_percentage_ = 1.0 * right_indices.size()/indices.size();
            node->right_child_ = right_node;
        }
    }
    else
    {
        this->setLeafNode(features, labels, indices, node);
    }
    return true;
}


bool BTDTCTree::bestSplitParameter(const vector<VectorXf> & features,
                                   const vector<int> & labels,
                                   const vector<int> & indices,
                                   const int depth,
                                   SplitParameter & split_param,
                                   vector<int> & left_indices,
                                   vector<int> & right_indices)
                              
{
    // randomly select number in a range
    const int dim = split_param.split_dim_;
    const int category_num = tree_param_.category_num_;
    const int threshold_num = tree_param_.candidate_threshold_num_;
    const int min_split_sample_num = tree_param_.min_split_size_;
    
    // step 1: variable range
    double min_v = std::numeric_limits<double>::max();
    double max_v = std::numeric_limits<double>::min();
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
    vector<double> split_values = DTRandom::generateRandomNumber(min_v, max_v, threshold_num);
    
    // step 2: optimize split value
    bool use_balance = (depth <= tree_param_.max_balanced_depth_);
    bool is_split = false;
    double loss = std::numeric_limits<double>::max();
    for (int i = 0; i<split_values.size(); i++) {
        double threshold = split_values[i];
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
        
        if (cur_left_indices.size() < min_split_sample_num ||
            cur_right_indices.size() < min_split_sample_num) {
            //printf("lefe sample number %lu, right sample number %lu\n", cur_left_indices.size(), cur_right_indices.size());
            continue;
        }
        
        double cur_loss = 0.0;
        if (use_balance) {
            cur_loss += DTUtil::balanceLoss((int)cur_left_indices.size(), (int)cur_right_indices.size());
        } else {
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
            cur_loss = entropy;
        }
        
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


void BTDTCTree::setLeafNode(const vector<VectorXf> & features,
                            const vector<int> & labels,
                            const vector<int> & indices,
                            NodePtr node)
{
    assert(node);
    
    const int category_num  = tree_param_.category_num_;
    
    node->is_leaf_ = true;
    node->sample_num_ = (int)indices.size();
    node->sample_percentage_ = 1.0;
    
    // label distribution
    Eigen::VectorXf prob = Eigen::VectorXf::Zero(category_num);
    for (int i = 0; i<indices.size(); i++) {
        int label = labels[indices[i]];
        assert(label >= 0 && label < category_num);
        prob[label] += 1.0f;
    }
    prob /= indices.size();
    node->prob_ = prob;
    
    // mean feature
    node->feat_mean_ = DTUtil::mean(features, indices);
    
    // count leaf node number
    leaf_node_num_++;
    
    if (tree_param_.verbose_leaf_) {
        printf("leaf node depth size %d    %lu\n", node->depth_, indices.size());
        cout<<"probability: \n"<<node->prob_.transpose()<<endl<<endl;;
    }
}


void BTDTCTree::hashLeafNode()
{
    assert(leaf_node_num_ > 0);
    leaf_nodes_.resize(leaf_node_num_);
    
    int index = 0;
    this->recordLeafNodes(root_, leaf_nodes_, index);
    printf("tree leaf node number is %d\n", leaf_node_num_);
}

void BTDTCTree::recordLeafNodes(const NodePtr node, vector<NodePtr> & leaf_nodes, int & index)
{
    assert(node);
    if (node->is_leaf_) {
        // for tree read from a file, index is precomputed
        if (node->index_ != -1) {
            assert(node->index_ == index);
        }
        node->index_ = index;
        leaf_nodes_[index] = node;
        index++;
        return;
    }
    if (node->left_child_) {
        this->recordLeafNodes(node->left_child_, leaf_nodes_, index);
    }
    if (node->right_child_) {
        this->recordLeafNodes(node->right_child_, leaf_nodes_, index);
    }
}

bool BTDTCTree::predict(const Eigen::VectorXf & feature,
                        const int max_check,
                        int & pred,
                        float & distance)
{
    Eigen::VectorXf prob;
    this->predict(feature, max_check, prob, distance);
    prob.maxCoeff(&pred);
    return true;
}

bool BTDTCTree::predict(const Eigen::VectorXf & feature,
                        const int max_check,
                        VectorXf & prob,
                        float & dist)
{
    assert(root_);
    
    //@todo check input feature dimension
    
    int check_count = 0;
    float eps_error = 1.0;
    const int knn = 1;
    
    BranchSt branch;
    flann::Heap<BranchSt> * heap = new flann::Heap<BranchSt>(leaf_node_num_);  // why use so large heap
    flann::DynamicBitset checked(leaf_node_num_);
    
    flann::KNNResultSet2<DistanceType> result(knn); // only keep the nearest one
    const ElementType *vec = feature.data();
    
    // search tree down to leaf
    this->searchLevel(result, vec, root_, 0, check_count, max_check, eps_error, heap, checked);
    
    while (heap->popMin(branch) &&
           (check_count < max_check || !result.full())) {
        assert(branch.node);
        this->searchLevel(result, vec, branch.node, branch.mindist, check_count, max_check, eps_error, heap, checked);
    }
    
    delete heap;
    assert(result.size() == knn);
    
    size_t index = 0;
    DistanceType distance;
    result.copy(&index, &distance, 1, false);
    
    prob = leaf_nodes_[index]->prob_;
    dist = (float)distance;
    

    return true;
}

void BTDTCTree::searchLevel(flann::ResultSet<DistanceType>  & result_set, const ElementType* vec, NodePtr node,
                            const DistanceType min_dist, int & check_count, const int max_check, const float eps_error,
                            flann::Heap<BranchSt>* heap, flann::DynamicBitset& checked) const
{
    if (result_set.worstDist() < min_dist) {
        return;
    }
    
    // check leaf node
    if (node->is_leaf_) {
        int index = node->index_;
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
    ElementType val = vec[node->split_param_.split_dim_];
    DistanceType diff = val - node->split_param_.split_threshold_;
    NodePtr bestChild  = (diff < 0 ) ? node->left_child_: node->right_child_;
    NodePtr otherChild = (diff < 0 ) ? node->right_child_: node->left_child_;
    
    DistanceType new_dist_sq = min_dist + distance_.accum_dist(val, node->split_param_.split_threshold_, node->split_param_.split_dim_);
    
    if ((new_dist_sq * eps_error < result_set.worstDist()) ||
        !result_set.full()) {
        heap->insert(BranchSt(otherChild, new_dist_sq));
    }
    
    // call recursively to search next level
    this->searchLevel(result_set, vec, bestChild, min_dist, check_count, max_check, eps_error, heap, checked);
}

bool BTDTCTree::writeTree(const char *fileName) const
{
    assert(root_);
    FILE *pf = fopen(fileName, "w");
    if (!pf) {
        printf("Error: can not open file %s\n", fileName);
        return false;
    }
    fprintf(pf, "%d %d\n", leaf_node_num_, tree_param_.category_num_);
    fprintf(pf, "depth\t isLeaf\t splitDim\t threshold\t percentage\t num\t probability\n");
    writeNode(pf, root_);
    fclose(pf);
    return true;
}

void BTDTCTree::writeNode(FILE *pf, NodePtr node)
{
    if (!node) {
        fprintf(pf, "#\n");
        return;
    }
    // write current node
    SplitParameter param = node->split_param_;
    fprintf(pf, "%2d\t %d\t %6d\t\t %lf\t %.2f\t\t %d\n",
            node->depth_, (int)node->is_leaf_,  param.split_dim_, param.split_threshold_, node->sample_percentage_, node->sample_num_);
    
    if (node->is_leaf_) {
        // leaf index and label distribution size
        fprintf(pf, "%d\n", node->index_);
        for (int i = 0; i<node->prob_.size(); i++) {
            fprintf(pf, "%lf ", node->prob_[i]);
        }
        fprintf(pf, "\n");
    }
    
    writeNode(pf, node->left_child_);
    writeNode(pf, node->right_child_);
}


bool BTDTCTree::readTree(const char *fileName)
{
    root_ = new Node(0);
    FILE *pf = fopen(fileName, "r");
    if (!pf) {
        printf("can not open file %s\n", fileName);
        return false;
    }
    
    int category_num = 0;
    // leaf node number
    int ret = fscanf(pf, "%d %d", &leaf_node_num_, &category_num);
    assert(ret == 2);
    
    // remove '\n' at the end of the line
    char dummy_line_buf[1024] = {NULL};
    fgets(dummy_line_buf, sizeof(dummy_line_buf), pf);
    
    //read marking line
    char line_buf[1024] = {NULL};
    fgets(line_buf, sizeof(line_buf), pf);
    printf("%s\n", line_buf);

    //read first line
    readNode(pf, root_, category_num);
    fclose(pf);
    
    return true;
}

void BTDTCTree::readNode(FILE *pf, NodePtr & node, const int category_num)
{
    assert(pf);
    char line_buf[1024] = {NULL};
    char *ret = fgets(line_buf, sizeof(line_buf), pf);
    if (!ret) {
        node = NULL;
        return;
    }
    if (line_buf[0] == '#') {
        // empty node
        node = NULL;
        return;
    }
    
    // read node parameters
    node = new Node(0);
    assert(node);
    int depth = 0;
    int is_leaf = 0;
    int split_dim = 0;
    double split_threshold = 0.0;
    double sample_percentage = 0.0;
    int sample_num = 0;
    
    int ret_num = sscanf(line_buf, "%d %d %d %lf %lf %d",
                         &depth, &is_leaf, &split_dim, &split_threshold, &sample_percentage, &sample_num);
    assert(ret_num == 6);
    
    node->depth_ = depth;
    node->is_leaf_ = is_leaf;
    node->sample_num_ = sample_num;
    node->sample_percentage_ = sample_percentage;
    
    node->split_param_.split_dim_ = split_dim;
    node->split_param_.split_threshold_ = split_threshold;
    
    if (is_leaf) {
        int leaf_node_index = 0;
        ret_num = fscanf(pf, "%d", &leaf_node_index);
        assert(ret_num == 1);
        node->index_ = leaf_node_index;
        
        Eigen::VectorXf prob = Eigen::VectorXf::Zero(category_num);
        for (int i = 0; i<category_num; i++) {
            double val = 0;
            ret_num = fscanf(pf, "%lf", &val);
            assert(ret_num);
            prob[i] = val;
        }
        // remove '\n' at the end of the line
        char dummy_line_buf[1024] = {NULL};
        fgets(dummy_line_buf, sizeof(dummy_line_buf), pf);
        node->prob_ = prob;
    }
    readNode(pf, node->left_child_, category_num);
    readNode(pf, node->right_child_, category_num);
    
}

