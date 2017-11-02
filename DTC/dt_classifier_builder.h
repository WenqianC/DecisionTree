//
//  DTClassifierBuilder.h
//  Classifer_RF
//
//  Created by jimmy on 2016-09-20.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef __Classifer_RF__DTClassifierBuilder__
#define __Classifer_RF__DTClassifierBuilder__

#include <stdio.h>
#include <Eigen/Dense>
#include "dt_classifier.h"

using Eigen::VectorXf;

class DTClassifierBuilder
{
private:
    
    DTCTreeParameter tree_param_;
    
public:
    void setTreeParameter(const DTCTreeParameter & param);
    
    // valid --> validation
    bool buildModel(DTClassifier & model,
                    const vector<VectorXf> & features,
                    const vector<int> & labels,
                    const vector<VectorXf> & validation_features,
                    const vector<int>& validation_labels,
                    const char * model_file_name = NULL) const;
    
    //features: a group of features, each group is from a single image
    //labels  : corresponding label
    bool buildModel(DTClassifier & model,
                    const vector< vector<VectorXf> > & features,
                    const vector< vector<int> > & labels,
                    const int max_num_frames,
                    const char * model_file_name = NULL) const;
    
};

#endif /* defined(__Classifer_RF__DTClassifierBuilder__) */
