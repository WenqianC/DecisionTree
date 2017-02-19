//
//  tp_dt_classifier_builder.h
//  Classifer_RF
//
//  Created by jimmy on 2017-02-19.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __Classifer_RF__tp_dt_classifier_builder__
#define __Classifer_RF__tp_dt_classifier_builder__

#include <stdio.h>
#include "tp_dt_classifier.h"

class TPDTClassifierBuilder
{
private:    
    
    TPDTCTreeParameter tree_param_;
    
public:
    void setTreeParameter(const TPDTCTreeParameter & param);
    
    bool buildModel(TPDTClassifier & model,
                    const vector<MatrixXf> & features,
                    const vector<unsigned int> & labels,
                    const char * model_file_name = NULL) const;
};


#endif /* defined(__Classifer_RF__tp_dt_classifier_builder__) */
