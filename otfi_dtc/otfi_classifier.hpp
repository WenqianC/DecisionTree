//
//  otfi_classifier.h
//  SequentialRandomForest
//
//  Created by jimmy on 2017-07-17.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __SequentialRandomForest__otfi_classifier__
#define __SequentialRandomForest__otfi_classifier__

// on the fly imputation (for classification)
// The goal is to imputate missing data in the feature
#include <stdio.h>
#include <Eigen/Dense>
#include <vector>
#include "otfi_util.hpp"
#include "otfi_tree_node.hpp"
#include "otfi_tree.hpp"

using std::vector;

class OTFIClassifier {
    friend class OTFIClassifierBuilder;
    
    typedef OTFITreeNode Node;
    typedef OTFITreeNode* NodePtr;
    typedef OTFITree Tree;
    typedef OTFITree* TreePtr;
    
    typedef OTFITreeParameter TreeParameter;
    
    vector<TreePtr> trees_;
    TreeParameter tree_param_;    
    
    int feature_dim_;         // feature dimension    

    public:
    
    OTFIClassifier();
    ~OTFIClassifier();
    
    bool predict(const Eigen::VectorXf & feature,
                 int & prediction) const;

    // impute missing data
    // mdata_features: missing data feature,
    // mdata_mask: missing data mask,
    bool imputeFeature(const vector<Eigen::VectorXf> & features,
                       const vector<int> & labels,                       
                       vector<Eigen::VectorXf> & mdata_features,  // output
                       const vector<int> & mdata_labels,                       
                       const float mdata_mask) const;
    
    bool save(const char *fileName) const;
    bool load(const char *fileName);

};

#endif /* defined(__SequentialRandomForest__otfi_classifier__) */
