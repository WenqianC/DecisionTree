//  Created by jimmy on 2016-12-30.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef __BT_DT_Regressor_Builder__
#define __BT_DT_Regressor_Builder__

#include <stdio.h>
#include "bt_dt_regressor.h"

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
    //maxCheck: leaf number in backtracking
    bool buildModel(BTDTRegressor & model,
                    const vector< vector<VectorXf> > & features,
                    const vector< vector<VectorXf> > & labels,
                    const int max_num_frames,
                    const int maxCheck,
                    const char * model_file_name = NULL) const;
    
    //features: a group of features, each group is from a single image
    //labels  : corresponding label
    //maxCheck: leaf number in backtracking
    //boostingRatio: percentage of re-training frames that are
    //               directly from cross validation test
    bool buildModel(BTDTRegressor & model,
                    const vector< vector<VectorXf> > & features,
                    const vector< vector<VectorXf> > & labels,
                    const int sampleFrameNum,
                    const int maxCheck,
                    const float boostingRatio,
                    const char * model_file_name = NULL) const;
    
};


#endif /* defined(__RGBD_RF__BTDTRegressorBuilder__) */
