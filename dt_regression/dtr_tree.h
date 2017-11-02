//
//  dtr_tree.h
//  SequentialRandomForest
//
//  Created by jimmy on 2017-11-01.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __SequentialRandomForest__dtr_tree__
#define __SequentialRandomForest__dtr_tree__

#include <stdio.h>
#include <vector>
#include <Eigen/Dense>
#include "dtr_param.h"
#include "dt_random.hpp"


using std::vector;
using Eigen::VectorXf;
using Eigen::VectorXi;


// decision tree regression Tree
class DTRTree
{
    friend class DTRegressor;
    
    typedef DTRSplitParameter SplitParameter;
    typedef DTRTreeParameter  TreeParameter;
    
    // internal data structures
    struct Node
    {
        Node* left_child_;
        Node* right_child_;
        int depth_;
        bool is_leaf_;
        
        SplitParameter split_param_;  // split parameter
        int sample_num_;
        double sample_percentage_;    // sample percentage of parent node
        
        VectorXf mean_;  // label mean, leaf node
        VectorXf stddev_;
        
        Node(int depth) {
            left_child_ = NULL;
            right_child_ = NULL;
            depth_ = depth;
            is_leaf_ = false;
            
            sample_num_ = 0;
            sample_percentage_ = 0.0;
        }
        ~Node() {
            if (left_child_) {
                delete left_child_;
                left_child_ = NULL;
            }
            if (right_child_) {
                delete right_child_;
                right_child_ = NULL;
            }
        }
    };
    
    typedef Node* NodePtr;
    
    NodePtr root_;
    TreeParameter tree_param_;
    DTRandom rnd_generator_;
    
public:
    DTRTree(){root_ = NULL;}
    ~DTRTree();
    
    
    // features:
    // labels: 0 - N-1
    // indices:
    bool buildTree(const vector<VectorXf> & features,
                   const vector<VectorXf> & labels,
                   const vector<int> & indices,
                   const TreeParameter & param);
    
    bool predict(const Eigen::VectorXf & feature,
                 Eigen::VectorXf & pred) const;
    
    
    const TreeParameter & getTreeParameter(void) const;
    void setTreeParameter(const TreeParameter & param);
    
    
private:
    bool buildTreeImpl(const vector<VectorXf> & features,
                       const vector<VectorXf> & labels,
                       const vector<int> & indices,
                       NodePtr node);
    
    bool setLeafNode(const vector<VectorXf> & features,
                     const vector<VectorXf> & labels,
                     const vector<int> & indices,
                     NodePtr node);
    
    bool bestSplitParameter(const vector<VectorXf> & features,
                            const vector<VectorXf> & labels,
                            const vector<int> & indices,
                            SplitParameter & split_param,
                            vector<int> & left_indices,
                            vector<int> & right_indices);
    
    bool predict(const NodePtr node,
                 const Eigen::VectorXf & feature,
                 Eigen::VectorXf & pred) const;
    
public:
    
    // read/write
    bool writeTree(const char *fileName) const;
    bool readTree(const char *fileName);
    
private:
    static void writeNode(FILE *pf, const NodePtr node);
    static void readNode(FILE *pf, NodePtr & node);
    
    
};


#endif /* defined(__SequentialRandomForest__dtr_tree__) */
