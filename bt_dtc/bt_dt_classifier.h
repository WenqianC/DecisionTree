//
//  bt_dt_classifier.h
//  PTZBTRF
//
//  Created by jimmy on 2017-10-13.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __PTZBTRF__bt_dt_classifier__
#define __PTZBTRF__bt_dt_classifier__

#include <stdio.h>
#include <vector>
#include "bt_dtc_tree.h"

using std::vector;

class BTDTClassifier
{
    typedef BTDTCTree TreeType;
    typedef TreeType* TreePtr;
    typedef BTDTCTreeParameter  TreeParameter;
    
    vector<TreePtr> trees_;
    TreeParameter tree_param_;
    
    int feature_dim_;       // feature dimension
    int category_num_;
    
public:
    BTDTClassifier();
    ~BTDTClassifier();
    
    // max_check: backtracking number
    // predictions: prediction from each tree
    // dists: distance from corresponding leaf node
    bool predict(const Eigen::VectorXf & feature,
                 const int max_check,
                 vector<int> & predictions,
                 vector<float> & dists) const;
    
    bool save(const char *file_name) const;
    bool load(const char *file_name);
    
    
};

#endif /* defined(__PTZBTRF__bt_dt_classifier__) */
