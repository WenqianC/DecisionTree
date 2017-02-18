//
//  tp_dtr_regressor_builder.cpp
//  Classifer_RF
//
//  Created by jimmy on 2017-02-17.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "tp_dtr_regressor_builder.h"
#include "dt_util.h"
#include <iostream>

using std::cout;
using std::endl;

void TPDTRegressorBuilder::setTreeParameter(const TPDTRTreeParameter & param)
{
    tree_param_ = param;
}

bool TPDTRegressorBuilder::buildModel(TPDTRegressor & model,
                                      const vector<MatrixXf> & features,
                                      const vector<VectorXf> & labels,
                                      const char * model_file_name ) const
{
    model.reg_tree_param_ = tree_param_;
    model.trees_.clear();   // @todo release memory
    model.feature_channel_ = (int)features.front().rows();
    model.feature_dim_ = (int)features.front().cols();
    model.label_dim_ = (int)labels.front().size();
    
    const int tree_num = tree_param_.tree_num_;    
    for (int n = 0; n<tree_num; n++) {
        // bagging
        vector<unsigned int> training_indices;
        vector<unsigned int> validation_indices;
        DTRandom::outof_bag_sampling((unsigned int) features.size(), training_indices, validation_indices);
        
        TPDTRTree * tree = new TPDTRTree();
        assert(tree);
        double tt = clock();
        tree->buildTree(features, labels, training_indices, tree_param_);
        model.trees_.push_back(tree);
        printf("build tree %d cost %lf minutes\n", n, (clock()- tt)/CLOCKS_PER_SEC/60.0);
        
        // cross validation
        // test on the validation data
        vector<Eigen::VectorXf> cv_errors;
        for (int i = 0; i<validation_indices.size(); i++) {
            const int index = validation_indices[i];
            Eigen::VectorXf pred;
            bool is_pred = tree->predict(features[index], pred);
            if (is_pred) {
                cv_errors.push_back(pred - labels[index]);
            }
        }
        
        Eigen::VectorXf cv_mean_error;
        Eigen::VectorXf cv_median_error;
        DTUtil::meanMedianError<Eigen::VectorXf>(cv_errors, cv_mean_error, cv_median_error);
        cout<<"cross validation mean error: \n"<<cv_mean_error.transpose()<<"\n median error: \n"<<cv_median_error.transpose()<<endl;
        if (model_file_name != NULL) {
            model.save(model_file_name);
        }
    }
    
    return true;
}