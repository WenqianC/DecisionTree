//
//  tp_dtr_regressor.h
//  Classifer_RF
//
//  Created by jimmy on 2017-02-17.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __Classifer_RF__tp_dtr_regressor__
#define __Classifer_RF__tp_dtr_regressor__

#include <stdio.h>
#include <vector>
#include "tp_dtr_tree.h"

using std::vector;

class TPDTRegressor
{
public:
    friend class TPDTRegressorBuilder;
    friend class RDTBuilder;
    
    typedef TPDTRTreeNode Node;
    typedef TPDTRTreeNode* NodePtr;
    typedef TPDTRTree Tree;
    typedef TPDTRTree* TreePtr;
    
    vector<TreePtr> trees_;
    TPDTRTreeParameter reg_tree_param_;
    
    int feature_dim_;       // feature dimension
    int feature_channel_;   // e.g., from three cameras
    int label_dim_;
    
public:
    TPDTRegressor(){feature_dim_ = 0; label_dim_ = 0; label_dim_ = 0;}
    ~TPDTRegressor(){}
    
    bool predict(const Eigen::MatrixXf & feature,
                 Eigen::VectorXf & pred) const;   
    
    
    
    bool save(const char *fileName) const;
    bool load(const char *fileName);
    
};


#endif /* defined(__Classifer_RF__tp_dtr_regressor__) */
