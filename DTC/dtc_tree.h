//
//  DTCTree.h
//  Classifer_RF
//
//  Created by jimmy on 2016-07-26.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef __Classifer_RF__DTCTree__
#define __Classifer_RF__DTCTree__

#include <stdio.h>
#include <vector>
#include <Eigen/Dense>
#include "dtc_param.h"
#include "dt_random.hpp"


using std::vector;
using Eigen::VectorXf;
using Eigen::VectorXi;


// decision tree classifier Tree
class DTCTree
{
    friend class DTClassifier;
    
    typedef DTCSplitParameter SplitParameter;
    typedef DTCTreeParameter  TreeParameter;
    
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
        
        VectorXf prob_;  // label probability, leaf node
   
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
    DTCTree(){root_ = NULL;}
    ~DTCTree(){;}
    
    // features:
    // labels: 0 - N-1
    // indices:
    bool buildTree(const vector<VectorXf> & features,
                   const vector<int> & labels,
                   const vector<int> & indices,
                   const TreeParameter & param);
    
    bool predict(const Eigen::VectorXf & feature,
                 Eigen::VectorXf & prob) const;
    
    bool predict(const Eigen::VectorXf & feature,
                 int & pred) const;
    
    // analyze prediction using record the index of training examples
    // in the leaf node
    // train_ids: training example ids
    // train_ids_in_leaf: training example ids that reach the same leaf node.
    // It can be accumulative
    bool analyzePrediction(const vector<Eigen::VectorXf> & train_features,
                           const vector<int> & train_indices,
                           const Eigen::VectorXf & valid_feature,                           
                           Eigen::VectorXf & prob,
                           vector<int> & train_indices_in_leaf);
    
    const TreeParameter & getTreeParameter(void) const;
    void setTreeParameter(const TreeParameter & param);
    
    
private:
    bool buildTreeImpl(const vector<VectorXf> & features,
                       const vector<int> & labels,
                       const vector<int> & indices,
                       NodePtr node);
    
    bool setLeafNode(const vector<Eigen::VectorXf> & features,
                     const vector<int> & labels,
                     const vector<int> & indices,
                     NodePtr node);
    
    bool bestSplitParameter(const vector<VectorXf> & features,
                            const vector<int> & labels,
                            const vector<int> & indices,
                            DTCSplitParameter & split_param,
                            vector<int> & left_indices,
                            vector<int> & right_indices);
    
    bool predict(const NodePtr node,
                 const Eigen::VectorXf & feature,
                 Eigen::VectorXf & prob) const;
    
    
    bool analyzePredictionImpl(const NodePtr node,
                               const vector<Eigen::VectorXf> & train_features,                               
                               const vector<int> & train_indices,
                               const Eigen::VectorXf & valid_feature,
                               Eigen::VectorXf & prob,
                               vector<int> & train_indices_in_leaf);
                           
    
    
    // read/write
    bool writeTree(const char *fileName) const;
    bool readTree(const char *fileName);
    
    static void writeNode(FILE *pf, const NodePtr node);
    static void readNode(FILE *pf, NodePtr & node);
    
};


#endif /* defined(__Classifer_RF__DTCTree__) */
