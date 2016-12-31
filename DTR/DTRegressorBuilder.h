//
//  DTRegressorBuilder.h
//  Classifer_RF
//
//  Created by jimmy on 2016-10-04.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef __Classifer_RF__DTRegressorBuilder__
#define __Classifer_RF__DTRegressorBuilder__

#include <stdio.h>
#include "DTRegressor.h"

class DTRegressorBuilder
{
private:
    DTRTreeParameter tree_param_;
    
public:
    void setTreeParameter(const DTRTreeParameter & param);
    
    bool buildModel(DTRegressor & model,
                    const vector<VectorXd> & features,
                    const vector<VectorXd> & labels,
                    const char * model_file_name = NULL) const;
    
    //features: a group of features, each group is from a single image
    //labels  : corresponding label
    bool buildModel(DTRegressor & model,
                    const vector< vector<VectorXd> > & features,
                    const vector< vector<VectorXd> > & labels,
                    const int max_num_frames,
                    const char * model_file_name = NULL) const;    
};


#endif /* defined(__Classifer_RF__DTRegressorBuilder__) */
