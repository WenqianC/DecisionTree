//
//  tp_dtr_tree.h
//  Classifer_RF
//
//  Created by jimmy on 2017-02-16.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __Classifer_RF__tp_dtr_tree__
#define __Classifer_RF__tp_dtr_tree__

// trinary projection regression tree
#include <stdio.h>
#include "tp_dtr_util.h"
#include <Eigen/Dense>
#include <vector>
#include "DTRandom.h"

using Eigen::VectorXf;
using Eigen::MatrixXf;
using Eigen::MatrixXi;
using std::vector;

class TPDTRTreeNode;

class TPDTRTree
{
    friend class TPDTRegressor;
private:
    typedef TPDTRTreeNode Node;
    typedef TPDTRTreeNode* NodePtr;
    typedef TPDTRTreeParameter TreeParameter;
    
    typedef Eigen::MatrixXf FeatureType;
    typedef Eigen::VectorXf LabelType;
    
    NodePtr root_;
    TreeParameter tree_param_;
    
    vector<vector<int> > trinary_permutation_;  // permutation of feature combination
    DTRandom rnd_generator_;
    
public:
    TPDTRTree(){root_ = NULL;}
    ~TPDTRTree();
    
    
    // features: matrix, each row is a vector, multiple rows are from multiple input (e.g., cameras at the same time)
    // labels: regression label
    // indices:
    bool buildTree(const vector<MatrixXf> & features,
                   const vector<VectorXf> & labels,
                   const vector<unsigned int> & indices,
                   const TreeParameter & param);
    
    
    bool predict(const Eigen::MatrixXf & feature,
                 Eigen::VectorXf & pred) const;
    
    const TreeParameter & getTreeParameter(void) const;
    void setTreeParameter(const TreeParameter & param);
    
private:
    
    bool configureNode(const vector<MatrixXf> & features,
                       const vector<VectorXf> & labels,
                       const vector<unsigned int> & indices,
                       NodePtr node);
    
    bool setLeafNode(const vector<Eigen::MatrixXf> & features,
                     const vector<VectorXf> & labels,
                     const vector<unsigned int> & indices,
                     NodePtr node);
    
    bool bestSplitParameter(const vector<Eigen::MatrixXf> & features,
                            const vector<VectorXf> & labels,
                            const vector<unsigned int> & indices,
                            const TPDTRTreeParameter & tree_param,
                            const int depth,
                            TPDTRSplitParameter & split_param,
                            vector<unsigned int> & left_indices,
                            vector<unsigned int> & right_indices);
    
    bool predict(const NodePtr node,
                 const Eigen::MatrixXf & feature,
                 Eigen::VectorXf & pred) const;

    
    
    

    
};

#endif /* defined(__Classifer_RF__tp_dtr_tree__) */
