//
//  tp_dt_classifier_builder.cpp
//  Classifer_RF
//
//  Created by jimmy on 2017-02-19.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "tp_dt_classifier_builder.h"
#include "dt_util.h"
#include <iostream>

using std::cout;
using std::endl;

void TPDTClassifierBuilder::setTreeParameter(const TPDTCTreeParameter & param)
{
    tree_param_ = param;
}

bool TPDTClassifierBuilder::buildModel(TPDTClassifier & model,
                                       const vector<MatrixXf> & features,
                                       const vector<unsigned int> & labels,
                                       const char * model_file_name ) const
{
    model.reg_tree_param_ = tree_param_;
    model.trees_.clear();   // @todo release memory
    model.feature_channel_ = (int)features.front().rows();
    model.feature_dim_ = (int)features.front().cols();
    model.category_num_ = tree_param_.category_num_;
    
    const int tree_num = tree_param_.tree_num_;
    for (int n = 0; n<tree_num; n++) {
        // bagging
        vector<unsigned int> training_indices;
        vector<unsigned int> validation_indices;
        DTRandom::outof_bag_sampling((unsigned int) features.size(), training_indices, validation_indices);
        
        TPDTCTree * tree = new TPDTCTree();
        assert(tree);
        double tt = clock();
        tree->buildTree(features, labels, training_indices, tree_param_);
        model.trees_.push_back(tree);
        printf("build tree %d cost %lf minutes\n", n, (clock()- tt)/CLOCKS_PER_SEC/60.0);
        
        // cross validation
        // test on the validation data
        vector<unsigned int> cv_pred;
        vector<unsigned int> cv_gds;
        for (int i = 0; i<validation_indices.size(); i++) {
            const int index = validation_indices[i];
            unsigned int pred;
            bool is_pred = tree->predict(features[index], pred);
            if (is_pred) {
                cv_pred.push_back(pred);
                cv_gds.push_back(labels[index]);
            }
        }
        assert(cv_pred.size() == cv_gds.size());
       
        Eigen::MatrixXd cv_confus = DTUtil::confusionMatrix(cv_pred, cv_gds, tree_param_.category_num_, false);
        cout<<"cross validation confusion matrix: \n"<<cv_confus<<endl;
        
        if (model_file_name != NULL) {
            model.save(model_file_name);
        }
    }
    
    return true;
}

