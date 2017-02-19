//
//  tp_dtc_tree.h
//  Classifer_RF
//
//  Created by jimmy on 2017-02-19.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __Classifer_RF__tp_dtc_tree__
#define __Classifer_RF__tp_dtc_tree__

#include <stdio.h>
#include "tp_dtc_util.h"
#include <Eigen/Dense>
#include <vector>
#include "DTRandom.h"

using Eigen::VectorXf;
using Eigen::MatrixXf;
using Eigen::MatrixXi;
using std::vector;

class TPDTCTreeNode;

class TPDTCTree
{
    friend class TPDTClassifier;
private:
    typedef TPDTCTreeNode Node;
    typedef TPDTCTreeNode* NodePtr;
    typedef TPDTCTreeParameter TreeParameter;
    
    typedef Eigen::MatrixXf FeatureType;
    typedef unsigned int LabelType;
    
    NodePtr root_;
    TreeParameter tree_param_;
    
    vector<vector<int> > trinary_permutation_;  // permutation of feature combination
    DTRandom rnd_generator_;
    
public:
    TPDTCTree(){root_ = NULL;}
    ~TPDTCTree();
    
    
    // features: matrix, each row is a vector, multiple rows are from multiple input (e.g., cameras at the same time)
    // labels: classifcation label
    // indices:
    bool buildTree(const vector<MatrixXf> & features,
                   const vector<unsigned int> & labels,
                   const vector<unsigned int> & indices,
                   const TreeParameter & param);
    
    
    bool predict(const Eigen::MatrixXf & feature,
                 unsigned int & pred) const;
    
    bool predict(const Eigen::MatrixXf & feature,
                 Eigen::VectorXf & prob) const;
    
    const TreeParameter & getTreeParameter(void) const;
    void setTreeParameter(const TreeParameter & param);
    
private:
    
    bool configureNode(const vector<MatrixXf> & features,
                       const vector<unsigned int> & labels,
                       const vector<unsigned int> & indices,
                       NodePtr node);
    
    bool setLeafNode(const vector<Eigen::MatrixXf> & features,
                     const vector<unsigned int> & labels,
                     const vector<unsigned int> & indices,
                     NodePtr node);
    
    bool bestSplitParameter(const vector<Eigen::MatrixXf> & features,
                            const vector<unsigned int> & labels,
                            const vector<unsigned int> & indices,
                            const TPDTCTreeParameter & tree_param,
                            const int depth,
                            TPDTCSplitParameter & split_param,
                            vector<unsigned int> & left_indices,
                            vector<unsigned int> & right_indices);
    
    bool predict(const NodePtr node,
                 const Eigen::MatrixXf & feature,
                 Eigen::VectorXf & pred) const;
};


#endif /* defined(__Classifer_RF__tp_dtc_tree__) */
