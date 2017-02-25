//
//  tp_dt_classifier.h
//  Classifer_RF
//
//  Created by jimmy on 2017-02-19.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __Classifer_RF__tp_dt_classifier__
#define __Classifer_RF__tp_dt_classifier__

#include <stdio.h>
#include <vector>
#include "tp_dtc_tree.h"

using std::vector;

class TPDTClassifier
{
public:
    friend class TPDTClassifierBuilder;    
    
    typedef TPDTCTreeNode Node;
    typedef TPDTCTreeNode* NodePtr;
    typedef TPDTCTree Tree;
    typedef TPDTCTree* TreePtr;
    
    vector<TreePtr> trees_;
    TPDTCTreeParameter reg_tree_param_;
    
    int feature_dim_;       // feature dimension
    int feature_channel_;   // e.g., from three cameras
    int category_num_;
    
public:
    TPDTClassifier(){feature_dim_ = 0; feature_channel_ = 0; category_num_ = 0;}
    ~TPDTClassifier(){}
    
    bool predict(const Eigen::MatrixXf & feature,
                 unsigned int & pred) const;
    
    bool predict(const Eigen::MatrixXf & feature,
                 Eigen::VectorXf & prob) const;
    
    
    
    bool save(const char *fileName) const;
    bool load(const char *fileName);
    
};


#endif /* defined(__Classifer_RF__tp_dt_classifier__) */
