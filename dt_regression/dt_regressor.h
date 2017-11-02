//
//  dt_regressor.h
//  SequentialRandomForest
//
//  Created by jimmy on 2017-11-01.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __SequentialRandomForest__dt_regressor__
#define __SequentialRandomForest__dt_regressor__

#include <stdio.h>
#include <vector>
#include "dtr_tree.h"

using std::vector;

class DTRegressor
{
    friend class DTRegressorBuilder;
    
    typedef DTRTree TreeType;
    typedef TreeType* TreePtr;
    
    typedef DTRTreeParameter TreeParameter;
    
    vector<TreePtr> trees_;
    TreeParameter tree_param_;
    
public:
    DTRegressor();
    ~DTRegressor();
    
    bool predict(const Eigen::VectorXf & feature,
                 Eigen::VectorXf & pred) const;
    
    bool save(const char *fileName) const;
    bool load(const char *fileName);
};


#endif /* defined(__SequentialRandomForest__dt_regressor__) */
