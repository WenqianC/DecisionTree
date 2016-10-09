//
//  DTRegressorBuilder.cpp
//  Classifer_RF
//
//  Created by jimmy on 2016-10-04.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#include "DTRegressorBuilder.h"
#include "DTRandom.h"
#include <iostream>

using std::cout;
using std::endl;

void DTRegressorBuilder::setTreeParameter(const DTRTreeParameter & param)
{
    tree_param_ = param;
}

bool DTRegressorBuilder::buildModel(DTRegressor & model,
                                    const vector<VectorXd> & features,
                                    const vector<VectorXd> & labels,
                                    const char * model_file_name) const
{
    assert(features.size() == labels.size());
    
    model.reg_tree_param_ = tree_param_;
    model.trees_.clear();   // @todo release memory
    model.feature_dim_ = (int)features.front().size();
    model.label_dim_ = (int)labels.front().size();
    
    const int tree_num = tree_param_.tree_num_;
    for (int n = 0; n<tree_num; n++) {
        // bagging
        vector<unsigned int> training_indices;
        vector<unsigned int> validation_indices;
        DTRandom::outof_bag_sampling((unsigned int) features.size(), training_indices, validation_indices);
        
        DTRTree * tree = new DTRTree();
        assert(tree);
        double tt = clock();
        tree->buildTree(features, labels, training_indices, tree_param_);
        model.trees_.push_back(tree);
        
        printf("build tree %d cost %lf minutes\n", n, (clock()- tt)/CLOCKS_PER_SEC/60.0);
        // test on the validation data
        
        vector<Eigen::VectorXd> cv_errors;
        for (int i = 0; i<validation_indices.size(); i++) {
            const int index = validation_indices[i];
            Eigen::VectorXd pred;
            model.predict(features[index], pred);
            cv_errors.push_back(pred - labels[index]);
        }
        
        Eigen::VectorXd cv_mean_error;
        Eigen::VectorXd cv_median_error;
        DTRUtil::mean_median_error(cv_errors, cv_mean_error, cv_median_error);
        cout<<"cross validation mean error: "<<cv_mean_error<<" median error: "<<cv_median_error<<endl;
        if (model_file_name != NULL) {
            model.save(model_file_name);
        }
    }
    printf("build model done %lu trees.\n", model.trees_.size());

    return true;
}