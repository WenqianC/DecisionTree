//
//  seq_dt_regressor_builder.h
//  Classifer_RF
//
//  Created by jimmy on 2017-02-25.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __Classifer_RF__seq_dt_regressor_builder__
#define __Classifer_RF__seq_dt_regressor_builder__

#include <stdio.h>
#include "seq_dt_regressor.h"
#include <Eigen/Dense>



class SeqDTRegressorBuilder
{
private:
    SeqDTRTreeParameter tree_param_;
    
public:
    void setTreeParameter(const SeqDTRTreeParameter & param);
    
    // max_time_step: odd number
    bool buildModel(SeqDTRegressor & model,
                    const vector<int>& frame_numbers,
                    const vector<Eigen::VectorXf> & features,
                    const vector<Eigen::VectorXf> & labels,                   
                    const char * model_file_name = NULL) const;
};


#endif /* defined(__Classifer_RF__seq_dt_regressor_builder__) */
