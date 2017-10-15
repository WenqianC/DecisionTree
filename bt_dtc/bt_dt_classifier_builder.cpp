//
//  bt_dt_classifier_builder.cpp
//  PTZBTRF
//
//  Created by jimmy on 2017-10-13.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "bt_dt_classifier_builder.h"
#include <iostream>
#include "dt_util.hpp"

using std::cout;
using std::endl;

void BTDTClassifierBuilder::setTreeParameter(const TreeParameter & param)
{
    tree_param_ = param;
}

bool BTDTClassifierBuilder::buildModel(BTDTClassifier & model,
                                       const vector<VectorXf> & features,
                                       const vector<int> & labels,
                                       const vector<VectorXf> & valid_features,
                                       const vector<int>& valid_labels,
                                       const int max_check,
                                       const float distance_threshold,
                                       const char * model_file_name) const
                
{
    assert(features.size() == labels.size());
    assert(valid_features.size() == valid_labels.size());
    
    model.tree_param_ = tree_param_;
    model.feature_dim_ = (int)features[0].size();
    model.category_num_ = tree_param_.category_num_;
    model.trees_.clear();
    
    const int tree_num = tree_param_.tree_num_;
    const int category_num = tree_param_.category_num_;
    const int N = (int)features.size();
    
    for (int n = 0; n<tree_num; n++) {
        // bagging
        vector<int> training_indices;
        vector<int> validation_indices;
        DTRandom::outofBagSampling<int>(N, training_indices, validation_indices);
        
        TreePtr pTree = new TreeType();
        assert(pTree);
        double tt = clock();
        pTree->buildTree(features, labels, training_indices, tree_param_);
        model.trees_.push_back(pTree);
        
        printf("build tree %d cost %lf minutes\n", n, (clock()- tt)/CLOCKS_PER_SEC/60.0);
        
        
        // test on the validation data
        vector<int> cv_predictions;
        vector<int> cv_labels;
        int no_prediction_num = 0;
        for (int i = 0; i<validation_indices.size(); i++) {
            const int index = validation_indices[i];
            int pred = 0;
            bool is_pred = model.predict(features[index], max_check, pred, distance_threshold);
            if (is_pred) {
                cv_predictions.push_back(pred);
                cv_labels.push_back(labels[index]);
            }
            else {
                no_prediction_num++;
            }
        }
        
        Eigen::MatrixXd oob_conf = DTUtil::confusionMatrix(cv_predictions, cv_labels, category_num, false);
        printf("Distance threshold: %lf, no prediction ratio: %lf\n", distance_threshold, 1.0*no_prediction_num/validation_indices.size());
        cout<<"Out of bag validation confusion matrix: \n"<<oob_conf<<endl<<endl;
        if (model_file_name != NULL) {
            model.save(model_file_name);
        }
        
        if (valid_features.size() != 0) {
            vector<int> valid_predictions;
            vector<int> temp_valid_labels;
            int no_prediction_num = 0;
            for (int i = 0; i<valid_features.size(); i++) {
                int pred = 0;
                bool is_pred = model.predict(valid_features[i], max_check, pred, distance_threshold);
                if (is_pred) {
                    valid_predictions.push_back(pred);
                    temp_valid_labels.push_back(valid_labels[i]);
                }
                else {
                    no_prediction_num++;
                }
            }
            assert(valid_predictions.size() == temp_valid_labels.size());
            Eigen::MatrixXd valid_conf = DTUtil::confusionMatrix<int>(valid_predictions, temp_valid_labels, category_num, false);
            Eigen::VectorXd accuracy = DTUtil::accuracyFromConfusionMatrix(valid_conf);
            printf("Distance threshold: %lf, no prediction ratio: %lf\n", distance_threshold, 1.0*no_prediction_num/valid_features.size());
            cout<<"Validation confusion matrix: \n"<<valid_conf<<endl;
            cout<<"Validation precision: \n"<<accuracy.transpose()<<endl<<endl;
        }
    }
    return true;
}