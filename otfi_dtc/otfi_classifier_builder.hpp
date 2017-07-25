//
//  otfi_classifier_builder.h
//  SequentialRandomForest
//
//  Created by jimmy on 2017-07-17.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __SequentialRandomForest__otfi_classifier_builder__
#define __SequentialRandomForest__otfi_classifier_builder__

#include <stdio.h>
#include <vector>
#include <Eigen/Dense>
#include "otfi_classifier.hpp"
#include "otfi_util.hpp"


using std::vector;

class OTFIClassifierBuilder {
    typedef OTFITreeParameter TreeParameter;
    typedef OTFITree TreeType;
    typedef OTFITree* TreePtr;

    TreeParameter tree_param_;
    
    public:
    OTFIClassifierBuilder();
    ~OTFIClassifierBuilder();

    void setTreeParameter(const TreeParameter & param);
    void buildModel(OTFIClassifier & model, 
                    const vector<Eigen::VectorXf> & features,
                    const vector<int> & labels);


};

#endif /* defined(__SequentialRandomForest__otfi_classifier_builder__) */
