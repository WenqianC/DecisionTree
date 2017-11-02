//
//  dt_regressor_builder.h
//  SequentialRandomForest
//
//  Created by jimmy on 2017-11-02.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __SequentialRandomForest__dt_regressor_builder__
#define __SequentialRandomForest__dt_regressor_builder__

#include <stdio.h>
#include <Eigen/Dense>
#include "dt_regressor.h"
#include "dtr_param.h"

using Eigen::VectorXf;

class DTRegressorBuilder
{    
    typedef DTRTree TreeType;
    typedef TreeType* TreePtr;
    
    typedef DTRTreeParameter TreeParameter;
    
private:
    
    TreeParameter tree_param_;
    
public:
    void setTreeParameter(const TreeParameter & param);
    
    // valid --> validation
    bool buildModel(DTRegressor & model,
                    const vector<VectorXf> & features,
                    const vector<VectorXf> & labels,
                    const vector<VectorXf> & validation_features,
                    const vector<VectorXf>& validation_labels,
                    const char * model_file_name = NULL) const;
};


#endif /* defined(__SequentialRandomForest__dt_regressor_builder__) */
