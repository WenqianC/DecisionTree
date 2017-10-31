//
//  DTClassifier.h
//  Classifer_RF
//
//  Created by jimmy on 2016-09-20.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef __Classifier_RF__DTClassifier__
#define __Classifier_RF__DTClassifier__

#include <stdio.h>
#include <vector>
#include "dtc_tree.h"

using std::vector;

class DTClassifier
{
    friend class DTClassifierBuilder;
    
    vector<DTCTree* > trees_;
    DTCTreeParameter tree_param_;
    
public:
    bool predict(const Eigen::VectorXf & feature,
                 Eigen::VectorXf & prob) const;
    
    bool predict(const Eigen::VectorXf & feature,
                 int & pred);
    
    // measure variable importance
    // return: accuracy decresment of one variable is replaced by random values
    Eigen::VectorXd measureVariableImportance(const vector<Eigen::VectorXf> & features,
                                              const vector<int> & labels);
    
    // get contritueing training example indices
    // simulate_oob_sampling: simulate out of bagging sample, default true
    // pred: predicted label
    // contributing_example_index: may have duplicated indices
    bool getContributingExamples(const vector<Eigen::VectorXf> & training_features,
                                 const Eigen::VectorXf & valid_feature,
                                 const bool simulate_oob_sampling,
                                 int & pred,
                                 vector<int> & contributing_example_index);
    
    bool save(const char *fileName) const;
    bool load(const char *fileName);
};

#endif /* defined(__Classifer_RF__DTClassifier__) */
