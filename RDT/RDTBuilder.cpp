//
//  RDTBuilder.cpp
//  Classifer_RF
//
//  Created by jimmy on 2016-10-07.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#include "RDTBuilder.h"
#include "DTRegressorBuilder.h"
#include <iostream>

using std::cout;
using std::endl;

RDTBuilder::RDTBuilder()
{
    
}
RDTBuilder::~RDTBuilder()
{
    
}

bool RDTBuilder::buildModel(DTRegressor& model,
                            const vector<int> & fns,
                            const vector< Eigen::VectorXd > & inputs,
                            const vector< Eigen::VectorXd > & outputs,
                            const SeqFeatGenerator & feature_generator,
                            const RDTParameter & rdt_param,
                            const DTRTreeParameter & tree_param) const
{
    assert(fns.size() == inputs.size());
    assert(fns.size() == outputs.size());
    
    
    vector<Eigen::VectorXd> predicted_labels = outputs;  // initiate
    // iterately learning
    const int iter_num = rdt_param.iter_num_;
    const double decrease_ratio = rdt_param.decrease_ratio_;
    const string model_file = rdt_param.model_file_;
    double alpha = 1.0;
    for (int n = 0; n < iter_num; n++) {
        alpha = alpha * decrease_ratio;
    
        printf("alpha is %f\n", alpha);
        
        // collect new feature
        vector<Eigen::VectorXd > rdt_features;
        vector<Eigen::VectorXd> rdt_labels;
        vector<int> original_fn_index;
        
        feature_generator.generateFeatures(predicted_labels, alpha, rdt_features, rdt_labels, original_fn_index);
        assert(rdt_features.size() == rdt_labels.size());
        
        // train the model
        double tt = clock();
        DTRegressorBuilder builder;
        builder.setTreeParameter(tree_param);
        
        builder.buildModel(model, rdt_features, rdt_labels, NULL);
        printf("training model cost time: %f minutes.\n", (clock() - tt)/CLOCKS_PER_SEC/60.0);
        // predict by current regressor
        vector<Eigen::VectorXd> diffs;
        for (int i = 0; i<rdt_features.size(); i++) {
            Eigen::VectorXd pred;
            bool is_pred = model.predict(rdt_features[i], pred);
            assert(is_pred);
            predicted_labels[original_fn_index[i]] = pred; //update predicted value
            diffs.push_back(pred - rdt_labels[i]);
        }
        
        Eigen::VectorXd cv_mean_error;
        Eigen::VectorXd cv_median_error;
        DTRUtil::mean_median_error(diffs, cv_mean_error, cv_median_error);
        cout<<"RDT mean error: "<<cv_mean_error<<" median error: "<<cv_median_error<<endl;
        
        if (model_file.length() != 0) {
            char buf[1024] = {NULL};
            string base = model_file.substr(0, model_file.size()-4);
            sprintf(buf, "%s_iter_%d.txt", base.c_str(), n);
            bool is_save = model.save(buf);
            if (!is_save) {
                printf("Error: can not save to %s.\n", buf);
            }
        }
    }
    if (model_file.length() != 0) {
        bool is_save = model.save(model_file.c_str());
        if (!is_save) {
            printf("Error: can not save to %s.\n", model_file.c_str());
        }        
    }
    return true;
}