//
//  tp_dtr_regressor_builder.h
//  Classifer_RF
//
//  Created by jimmy on 2017-02-17.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __Classifer_RF__tp_dtr_regressor_builder__
#define __Classifer_RF__tp_dtr_regressor_builder__

#include <stdio.h>
#include "tp_dtr_regressor.h"

class TPDTRegressorBuilder
{
private:
    TPDTRTreeParameter tree_param_;
    
public:
    void setTreeParameter(const TPDTRTreeParameter & param);
    
    bool buildModel(TPDTRegressor & model,
                    const vector<MatrixXf> & features,
                    const vector<VectorXf> & labels,
                    const char * model_file_name = NULL) const;
};


#endif /* defined(__Classifer_RF__tp_dtr_regressor_builder__) */
