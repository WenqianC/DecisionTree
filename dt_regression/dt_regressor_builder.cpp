//
//  dt_regressor_builder.cpp
//  SequentialRandomForest
//
//  Created by jimmy on 2017-11-02.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "dt_regressor_builder.h"
#include "dt_random.hpp"
#include <iostream>
#include "dt_util.hpp"


using std::cout;
using std::endl;

void DTRegressorBuilder::setTreeParameter(const TreeParameter & param)
{
    tree_param_ = param;
}

bool DTRegressorBuilder::buildModel(DTRegressor & model,
                                     const vector<VectorXf> & features,
                                     const vector<VectorXf> & labels,
                                     const vector<VectorXf> & validation_features,
                                     const vector<VectorXf> & validation_labels,
                                     const char * model_file_name) const
{
    assert(features.size() == labels.size());
    assert(validation_features.size() == validation_labels.size());
    
    model.tree_param_ = tree_param_;
    model.trees_.clear();
    const int tree_num = tree_param_.tree_num_;
    const int N = (int)features.size();
    
    DTRandom rng;
    for (int n = 0; n<tree_num; n++) {
        // bagging
        vector<int> training_indices;
        vector<int> oob_indices;
        rng.outofBagSample<int>(N, training_indices, oob_indices);
        
        TreePtr pTree = new TreeType();
        assert(pTree);
        double tt = clock();
        pTree->buildTree(features, labels, training_indices, tree_param_);
        model.trees_.push_back(pTree);
        
        printf("build tree %d cost %lf minutes\n", n, (clock()- tt)/CLOCKS_PER_SEC/60.0);
        
        
        // test on the oob data
        {
            vector<Eigen::VectorXf> oob_errors;
            for (int i = 0; i<oob_indices.size(); i++) {
                int index = oob_indices[i];
                Eigen::VectorXf pred;
                bool is_pred = pTree->predict(features[index], pred);
                if (is_pred) {
                    Eigen::VectorXf dif = pred - labels[index];
                    oob_errors.push_back(dif);
                }
            }
            
            Eigen::VectorXf mean_error;
            Eigen::VectorXf median_error;
            dt::meanMedianError(oob_errors, mean_error, median_error);
            cout<<"Out of bag mean error: "<<mean_error.transpose()<<"\nmedian error: "<<median_error.transpose()<<endl<<endl;
        }
        
        if (validation_features.size() != 0) {
            vector<Eigen::VectorXf> errors;
            for (int i = 0; i<validation_features.size(); i++) {
                Eigen::VectorXf pred;
                bool is_pred = model.predict(validation_features[i], pred);
                if (is_pred) {
                    Eigen::VectorXf dif = pred - validation_labels[i];
                    errors.push_back(dif);
                }
            }
            assert(validation_features.size() == validation_labels.size());
            Eigen::VectorXf mean_error;
            Eigen::VectorXf median_error;
            dt::meanMedianError(errors, mean_error, median_error);
            cout<<"Validation mean error: "<<mean_error.transpose()<<"\nmedian error: "<<median_error.transpose()<<endl<<endl;
        }
        if (model_file_name != NULL) {
            model.save(model_file_name);
        }        
    }
    return true;
}


