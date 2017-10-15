//
//  bt_dt_classifier_builder.h
//  PTZBTRF
//
//  Created by jimmy on 2017-10-13.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __PTZBTRF__bt_dt_classifier_builder__
#define __PTZBTRF__bt_dt_classifier_builder__

#include <stdio.h>
#include "bt_dt_classifier.h"

class BTDTClassifierBuilder
{
    typedef BTDTCTree TreeType;
    typedef TreeType* TreePtr;
    typedef BTDTCTreeParameter TreeParameter;
    
    TreeParameter tree_param_;
    
public:
    void setTreeParameter(const TreeParameter & param);
    
    bool buildModel(BTDTClassifier & model,
                    const vector<VectorXf> & features,
                    const vector<int> & labels,
                    const vector<VectorXf> & valid_features,
                    const vector<int>& valid_labels,
                    const int max_check,
                    const float distance_threshold = INT_MAX,
                    const char * model_file_name = NULL) const;

    
};

#endif /* defined(__PTZBTRF__bt_dt_classifier_builder__) */
