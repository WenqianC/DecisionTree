//
//  DTCTree.cpp
//  Classifer_RF
//
//  Created by jimmy on 2016-07-26.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#include "dtc_tree.h"
#include <algorithm>
#include <iostream>
#include "dt_util.hpp"

using std::cout;
using std::endl;

bool DTCTree::buildTree(const vector<VectorXf> & features,
                        const vector<int> & labels,
                        const vector<int> & indices,
                        const DTCTreeParameter & param)
{
    assert(features.size() == labels.size());
    assert(indices.size() <= features.size());
    
    tree_param_ = param;
    root_ = new Node(0);
    
    return this->buildTreeImpl(features, labels, indices, root_);
}



bool DTCTree::buildTreeImpl(const vector<VectorXf> & features,
                            const vector<int> & labels,
                            const vector<int> & indices,
                            NodePtr node)
{
    assert(node);
    const int min_leaf_node = tree_param_.min_leaf_node_num_;
    const int max_depth     = tree_param_.max_depth_;
    
    const int depth = node->depth_;
    const int dim = (int)features[0].size();
    
    // leaf node
    if (indices.size() < min_leaf_node || depth > max_depth ||
        ( depth > max_depth/2 && DTUtil::isSameLabel(labels, indices))) {
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

bool DTCTree::setLeafNode(const vector<Eigen::VectorXf> & features,
                          const vector<int> & labels,
                          const vector<int> & indices,
                          NodePtr node)
{
    assert(node);
    const int category_num  = tree_param_.category_num_;
    
    node->is_leaf_ = true;
    Eigen::VectorXf prob = Eigen::VectorXf::Zero(category_num);
    for (int i = 0; i<indices.size(); i++) {
        assert(labels[indices[i]] >= 0 && labels[indices[i]] < category_num);
        prob[labels[indices[i]]] += 1.0f;
    }
    prob /= indices.size();
    node->prob_ = prob;
    node->sample_num_ = (int)indices.size();
    if (tree_param_.verbose_leaf_) {
        printf("leaf node depth size %d    %lu\n", node->depth_, indices.size());
        cout<<"probability: \n"<<node->prob_.transpose()<<endl<<endl;;
    }    
    return true;
}

bool DTCTree::bestSplitParameter(const vector<VectorXf> & features,
                                 const vector<int> & labels,
                                 const vector<int> & indices,
                                 DTCSplitParameter & split_param,
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
    const int category_num  = tree_param_.category_num_;
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


bool DTCTree::predict(const Eigen::VectorXf & feature,
                      Eigen::VectorXf & prob) const
{
    assert(feature.size() == tree_param_.feature_dimension_);
    assert(root_);
    
    return this->predict(root_, feature, prob);
}

bool DTCTree::predict(const Eigen::VectorXf & feature,
                      unsigned int & pred) const
{
    Eigen::VectorXf prob;
    bool isPred = this->predict(feature, prob);
    if (!isPred) {
        return false;
    }
    prob.maxCoeff(&pred);
    return true;    
}


const DTCTree::TreeParameter & DTCTree::getTreeParameter(void) const
{
    return tree_param_;
}

void DTCTree::setTreeParameter(const TreeParameter & param)
{
    tree_param_ = param;
}


bool DTCTree::predict(const NodePtr node,
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

// read/write tree
void DTCTree::writeNode(FILE *pf, NodePtr node)
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
        fprintf(pf, "%d\n", (int)node->prob_.size());
        for (int i = 0; i<node->prob_.size(); i++) {
            fprintf(pf, "%lf ", node->prob_[i]);
        }
        fprintf(pf, "\n");
    }
    
    writeNode(pf, node->left_child_);
    writeNode(pf, node->right_child_);
}


bool DTCTree::writeTree(const char *fileName) const
{
    assert(root_);
    FILE *pf = fopen(fileName, "w");
    if (!pf) {
        printf("Error: can not open file %s\n", fileName);
        return false;
    }
    fprintf(pf, "depth\t isLeaf\t splitDim\t threshold\t percentage\t num\t probability\n");
    writeNode(pf, root_);
    fclose(pf);
    return true;
}



void DTCTree::readNode(FILE *pf, NodePtr & node)
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
        Eigen::VectorXf prob = Eigen::VectorXf::Zero(label_dim);
        for (int i = 0; i<label_dim; i++) {
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
    readNode(pf, node->left_child_);
    readNode(pf, node->right_child_);
}

bool DTCTree::readTree(const char *fileName)
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


