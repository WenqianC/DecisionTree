//
//  otfi_tree.h
//  SequentialRandomForest
//
//  Created by jimmy on 2017-07-16.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __SequentialRandomForest__otfi_tree__
#define __SequentialRandomForest__otfi_tree__

// on the fly imputation tree
// From Random survival forests 2008

#include <stdio.h>
#include <Eigen/Dense>

#include "otfi_util.hpp"
#include "dt_random.hpp"
#include "otfi_tree_node.hpp"

using Eigen::VectorXf;

class OTFITree
{
    friend class OTFIClassifier;
    
    typedef OTFITreeNode Node;
    typedef OTFITreeNode* NodePtr;
    typedef OTFITreeParameter TreeParameter;
    typedef OTFISplitParameter SplitParameter;
    
private:
    NodePtr root_;
    TreeParameter tree_param_;
    DTRandom rnd_generator_;    
    int feature_dims_;
    
public:
    OTFITree();
    ~OTFITree();

    const OTFITree::TreeParameter & getTreeParameter(void) const;
    void setTreeParameter(const TreeParameter & param);
    
    // build a decision tree using training examples
    // features: sampled image pixel locations
    // labels: 3D location
    // indices: index of samples
    // missing_mask: missing mask, feature with this value is a mask of missed value
    // param: tree parameter
    bool buildTree(const vector<Eigen::VectorXf> & features,
                   const vector<int> & labels,
                   const vector<int> & indices,
                   const TreeParameter & param);
    
    // impute missing data
    // mdata_features: missing data feature,
    // mdata_mask: missing data mask,
    bool imputeFeature(const vector<Eigen::VectorXf> & features,
                       const vector<int> & labels,
                       const vector<int> & indices,
                       const vector<int> & mdata_labels,
                       const vector<int> & mdata_indices,
                       const float mdata_mask,
                       vector<Eigen::VectorXf> & mdata_features,  // output
                       vector<float> & weight) const; // output
        
    bool predict(const Eigen::VectorXf & feature,
                 int & pred) const;

    bool predict(const Eigen::VectorXf & feature,
                 Eigen::VectorXf & prob) const;

    

private:
    
    bool buildTreeImpl(const vector<Eigen::VectorXf> & features,
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
                           SplitParameter & split_param,
                           vector<int> & left_indices,
                           vector<int> & right_indices);
    
    bool imputeFeatureImpl(const NodePtr node,
                const vector<Eigen::VectorXf> & features,
                const vector<int> & labels,
                const vector<int> & indices,               
                const vector<int> & mdata_labels,
                const vector<int> & mdata_indices,
                const float mdata_mask,
                vector<Eigen::VectorXf> & mdata_features,  // output
                vector<float> & weight) const; // output
    
    bool predictImpl(const NodePtr node,
                     const Eigen::VectorXf & feature,
                     Eigen::VectorXf & prob) const;
    
    
    
    
};

#endif /* defined(__SequentialRandomForest__otfi_tree__) */
