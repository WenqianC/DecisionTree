//
//  BTDTRegressorBuilder.h
//  RGBD_RF
//
//  Created by jimmy on 2016-12-30.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef __RGBD_RF__BTDTRegressorBuilder__
#define __RGBD_RF__BTDTRegressorBuilder__

#include <stdio.h>
#include "BTDTRegressor.h"

class BTDTRegressorBuilder
{
private:
    BTDTRTreeParameter tree_param_;
    
    typedef BTDTRTree TreeType;
    
public:
    void setTreeParameter(const BTDTRTreeParameter & param);
    
    bool buildModel(BTDTRegressor & model,
                    const vector<VectorXf> & features,
                    const vector<VectorXf> & labels,
                    const int maxCheck,
                    const char * model_file_name = NULL) const;
    
    //features: a group of features, each group is from a single image
    //labels  : corresponding label
    bool buildModel(BTDTRegressor & model,
                    const vector< vector<VectorXf> > & features,
                    const vector< vector<VectorXf> > & labels,
                    const int max_num_frames,
                    const int maxCheck,
                    const char * model_file_name = NULL) const;    
    
};


#endif /* defined(__RGBD_RF__BTDTRegressorBuilder__) */
