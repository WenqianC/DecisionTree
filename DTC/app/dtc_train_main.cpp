//
//  main.cpp
//  Classifer_RF
//
//  Created by jimmy on 2016-07-26.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#if 0

#include <iostream>
#include "dt_classifier.h"
#include "dt_classifier_builder.h"
#include "dt_util_io.hpp"
#include <Eigen/Dense>
#include "mat_io.hpp"

using Eigen::MatrixXf;
using Eigen::MatrixXi;

static void help()
{
    printf("program    featureFile labelFile DTParamFile saveFile\n");
    printf("DTC_train  feature.mat label.mat param.txt   dt_model.txt\n");
    printf("feature file: .mat file has a 'feature' variable. \n");
    printf("label file:   .mat file has a 'label' variable. \n");
}

static void readDataset(const char *feature_file,
                                            const char *label_file,
                                            vector<Eigen::VectorXf> & features,
                                            vector<int> & labels)
{
    Eigen::MatrixXf mat_features;
    Eigen::MatrixXi mat_labels;
    matio::readMatrix(feature_file, "feature", mat_features);
    matio::readMatrix(label_file, "label", mat_labels);
    assert(mat_labels.cols() == 1);
    
    for (int i = 0; i < mat_features.rows(); i++) {
        Eigen::VectorXf feat = mat_features.row(i);
        int label = mat_labels(i, 0);
        
        features.push_back(feat);
        labels.push_back(label);
    }
    assert(features.size() == labels.size());
    printf("read %lu train examples\n", features.size());
}


int main(int argc, const char * argv[])
{
    if (argc != 5) {
        printf("argc is %d, should be 5.\n", argc);
        help();
        return -1;
    }
    
    const char *feature_file = argv[1];
    const char *label_file = argv[2];
    const char *tree_param_file = argv[3];
    const char *save_file = argv[4];
    
    // read frame number, feature, label
    vector<VectorXf> features;
    vector<int> labels;
    vector<VectorXf> valid_features;
    vector<int>      valid_labels;
    readDataset(feature_file, label_file, features, labels);
    
   
    DTCTreeParameter tree_param;
    bool is_read = tree_param.readFromFile(tree_param_file);
    assert(is_read);
    
    tree_param.feature_dimension_ = (int)features[0].size();
    
    DTClassifierBuilder builder;
    builder.setTreeParameter(tree_param);
    
    DTClassifer model;
    builder.buildModel(model, features, labels, valid_features, valid_labels, save_file);
    
    model.save(save_file);
    printf("save model to %s\n", save_file);
    
    return 0;
}
#endif

