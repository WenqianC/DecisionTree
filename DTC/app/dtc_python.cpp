//
//  main.cpp
//  Classifer_RF
//
//  Created by jimmy on 2016-07-26.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#if 1


#include "dt_classifier.h"
#include "dt_classifier_builder.h"
#include "dt_util_io.hpp"
#include <iostream>
#include <Eigen/Dense>

using Eigen::MatrixXf;
using Eigen::MatrixXi;

extern "C" {
    void dtc_train(const void* input_feature,
                   const void* input_label,
                   int rows, int cols,
                   const char* param_file,
                   const char* save_file)
    {
       
        
        // read training data
        vector<VectorXf> features;
        vector<int> labels;
        
        Eigen::VectorXf feat(cols);
        const double* p_feature = (double*) input_feature;
        const double* p_label = (double*) input_label;
        for (int i = 0; i<rows; i++) {
            for (int j = 0; j<cols; j++) {
                feat[j] = p_feature[i*cols + j];
            }
            int label = p_label[i];
            features.push_back(feat);
            labels.push_back(label);
        }
        assert(features.size() == labels.size());
        printf("read %lu train examples\n", features.size());
        
        
        vector<VectorXf> valid_features;
        vector<int>      valid_labels;
        
        printf("tree parameter file name %s\n", param_file);
        printf("save file name %s\n", save_file);
        
        DTCTreeParameter tree_param;
        bool is_read = tree_param.readFromFile(param_file);
        assert(is_read);
        tree_param.feature_dimension_ = (int)features[0].size();
        const int thread_num = 4;
        
        DTClassifierBuilder builder;
        builder.setTreeParameter(tree_param);
        
        DTClassifier model;
        builder.buildModel(model, features, labels, valid_features, valid_labels, thread_num, save_file);
        
        model.save(save_file);
        printf("save model to %s\n", save_file);
    }    
}

#endif

