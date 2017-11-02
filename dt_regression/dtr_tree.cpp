//
//  dtr_tree.cpp
//  SequentialRandomForest
//
//  Created by jimmy on 2017-11-01.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "dtr_tree.h"
#include <algorithm>
#include <iostream>
#include "dt_util.hpp"

using std::cout;
using std::endl;

DTRTree::~DTRTree()
{
    if (root_) {
        delete root_;
        root_ = NULL;
    }
}

bool DTRTree::buildTree(const vector<VectorXf> & features,
                        const vector<VectorXf> & labels,
                        const vector<int> & indices,
                        const TreeParameter & param)
{
    assert(features.size() == labels.size());
    assert(indices.size() <= features.size());
    
    tree_param_ = param;
    root_ = new Node(0);
    
    return this->buildTreeImpl(features, labels, indices, root_);
}



bool DTRTree::buildTreeImpl(const vector<VectorXf> & features,
                            const vector<VectorXf> & labels,
                            const vector<int> & indices,
                            NodePtr node)
{
    assert(node);
    const int min_leaf_node = tree_param_.min_leaf_node_num_;
    const int max_depth     = tree_param_.max_depth_;
    
    const int depth = node->depth_;
    const int dim = (int)features[0].size();
    
    // leaf node
    if (indices.size() < min_leaf_node || depth > max_depth ) {
        this->setLeafNode(features, labels, indices, node);
        return true;
    }
    
    // randomly select a subset of dimensions
    vector<int> dims;
    for (int i = 0; i<dim; i++) {
        dims.push_back(i);
    }
    int sqrt_feat_dim = sqrt((double)dim);
    std::random_shuffle(dims.begin(), dims.end());
    vector<int> random_dim(dims.begin(), dims.begin() + sqrt_feat_dim);
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
        bool cur_is_split = bestSplitParameter(features, labels, indices, cur_split_param, cur_left_indices, cur_right_indices);
        if (cur_is_split && cur_split_param.loss_ < loss) {
            is_split = true;
            loss = cur_split_param.loss_;
            split_param = cur_split_param;
            left_indices = cur_left_indices;
            right_indices = cur_right_indices;
        }
    }
    
    // split data
    if (is_split) {
        assert(left_indices.size() + right_indices.size() == indices.size());
        if (tree_param_.verbose_) {
            printf("left node percentage: %f\n", 1.0 * left_indices.size()/indices.size());
            printf("cross entropy loss: %f\n", split_param.loss_);
        }
        node->split_param_ = split_param;
        node->is_leaf_ = false;
        node->sample_num_ = (int)indices.size();
        if (left_indices.size() > 0) {
            Node *left_node = new Node(depth + 1);
            this->buildTreeImpl(features, labels, left_indices, left_node);
            left_node->sample_percentage_ = 1.0 * left_indices.size()/indices.size();
            node->left_child_ = left_node;
        }
        if (right_indices.size() > 0) {
            Node * right_node = new Node(depth + 1);
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

bool DTRTree::setLeafNode(const vector<Eigen::VectorXf> & features,
                          const vector<VectorXf> & labels,
                          const vector<int> & indices,
                          NodePtr node)
{
    assert(node);
    dt::meanStd(labels, indices, node->mean_, node->stddev_);
    if (tree_param_.verbose_leaf_) {
        printf("leaf node depth size %d    %lu\n", node->depth_, indices.size());
        cout<<"mean  : \n"<<node->mean_<<endl;
        cout<<"stddev: \n"<<node->stddev_<<endl;
    }
    node->sample_num_ = (int)indices.size();
    node->is_leaf_ = true;
    return true;
}

bool DTRTree::bestSplitParameter(const vector<VectorXf> & features,
                                 const vector<VectorXf> & labels,
                                 const vector<int> & indices,
                                 SplitParameter & split_param,
                                 vector<int> & left_indices,
                                 vector<int> & right_indices)
{
    // randomly select number in a range
    const int dim = split_param.dim_;
    double min_v = std::numeric_limits<double>::max();
    double max_v = std::numeric_limits<double>::min();
    const int rand_num = tree_param_.split_candidate_num_;
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
    vector<double> rnd_split_values = rnd_generator_.getRandomNumbers(min_v, max_v, rand_num);
    
    bool is_split = false;
    double loss = std::numeric_limits<double>::max();
    const int min_split_num = tree_param_.min_split_num_;
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
        
        double cur_loss = 0.0;
        cur_loss += dt::sumOfVariance(labels, cur_left_indices);
        cur_loss += dt::sumOfVariance(labels, cur_right_indices);
        if (cur_loss < loss) {
            loss = cur_loss;
            is_split = true;
            left_indices = cur_left_indices;
            right_indices = cur_right_indices;
            split_param.threshold_ = threshold;
            split_param.loss_ = cur_loss;
        }
    }
    
    return is_split;
}


bool DTRTree::predict(const Eigen::VectorXf & feature,
                      Eigen::VectorXf & pred) const
{
    assert(feature.size() == tree_param_.feature_dimension_);
    assert(root_);
    
    return this->predict(root_, feature, pred);
}


const DTRTree::TreeParameter & DTRTree::getTreeParameter(void) const
{
    return tree_param_;
}

void DTRTree::setTreeParameter(const TreeParameter & param)
{
    tree_param_ = param;
}


bool DTRTree::predict(const NodePtr node,
                      const Eigen::VectorXf & feature,
                      Eigen::VectorXf & pred) const
{
    assert(node);
    if (node->is_leaf_) {
        pred = node->mean_;
        return true;
    }
    const double feat = feature[node->split_param_.dim_];
    if (feat < node->split_param_.threshold_ && node->left_child_) {
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
}


// read/write tree
void DTRTree::writeNode(FILE *pf, NodePtr node)
{
    if (!node) {
        fprintf(pf, "#\n");
        return;
    }
    // write current node
    SplitParameter param = node->split_param_;
    fprintf(pf, "%2d\t %d\t %6d\t\t %lf\t %.2f\t\t %d\n",
            node->depth_, (int)node->is_leaf_,  param.dim_, param.threshold_, node->sample_percentage_, node->sample_num_);
    
    if (node->is_leaf_) {
        fprintf(pf, "%d\n", (int)node->mean_.size());
        for (int i = 0; i<node->mean_.size(); i++) {
            fprintf(pf, "%lf ", node->mean_[i]);
        }
        fprintf(pf, "\n");
        for (int i = 0; i<node->stddev_.size(); i++) {
            fprintf(pf, "%lf ", node->stddev_[i]);
        }
        fprintf(pf, "\n");
    }
    
    writeNode(pf, node->left_child_);
    writeNode(pf, node->right_child_);
}


bool DTRTree::writeTree(const char *fileName) const
{
    assert(root_);
    FILE *pf = fopen(fileName, "w");
    if (!pf) {
        printf("Error: can not open file %s\n", fileName);
        return false;
    }
    fprintf(pf, "depth\t isLeaf\t splitDim\t threshold\t percentage\t num\t mean\t stdDev\n");
    writeNode(pf, root_);
    fclose(pf);
    return true;
}



void DTRTree::readNode(FILE *pf, NodePtr & node)
{
    char lineBuf[1024] = {NULL};
    char *ret = fgets(lineBuf, sizeof(lineBuf), pf);
    if (!ret) {
        node = NULL;
        return;
    }
    if (lineBuf[0] == '#') {
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
    
    int ret_num = sscanf(lineBuf, "%d %d %d %lf %lf %d",
                         &depth, &is_leaf, &split_dim, &split_threshold, &sample_percentage, &sample_num);
    assert(ret_num == 6);
    
    node->depth_ = depth;
    node->is_leaf_ = is_leaf;
    node->sample_num_ = sample_num;
    node->sample_percentage_ = sample_percentage;
    
    SplitParameter param;
    param.dim_ = split_dim;
    param.threshold_ = split_threshold;
    node->split_param_ = param;
    
    if (is_leaf) {
        int label_dim = 0;
        ret_num = fscanf(pf, "%d", &label_dim);
        assert(ret_num == 1);
        Eigen::VectorXf mean   = Eigen::VectorXf::Zero(label_dim);
        Eigen::VectorXf stddev = Eigen::VectorXf::Zero(label_dim);
        for (int i = 0; i<label_dim; i++) {
            double val = 0;
            ret_num = fscanf(pf, "%lf", &val);
            assert(ret_num);
            mean[i] = val;
        }
        for (int i = 0; i<label_dim; i++) {
            double val = 0;
            ret_num = fscanf(pf, "%lf", &val);
            assert(ret_num);
            stddev[i] = val;
        }
        // remove '\n' at the end of the line
        char dummy_line_buf[1024] = {NULL};
        fgets(dummy_line_buf, sizeof(dummy_line_buf), pf);
        node->mean_ = mean;
        node->stddev_ = stddev;
        
    }
    readNode(pf, node->left_child_);
    readNode(pf, node->right_child_);
}

bool DTRTree::readTree(const char *fileName)
{
    root_ = new Node(0);
    FILE *pf = fopen(fileName, "r");
    if (!pf) {
        printf("can not open file %s\n", fileName);
        return false;
    }
    //read first line
    char line_buf[1024] = {NULL};
    fgets(line_buf, sizeof(line_buf), pf);
    printf("%s\n", line_buf);
    readNode(pf, root_);
    fclose(pf);
    return true;
}



