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
#include "dtc_param.h"
#include <iostream>
#include <algorithm>
#include <fstream>
#include <Eigen/Dense>
#include "mat_io.hpp"
#include "dt_util.hpp"

using namespace::std;


static void help()
{
    printf("program                  featureFile labelFile modelFile     saveFile \n");
    printf("DTC_variable_importance  feature.mat label.mat dt_model.txt  accuracy_decresement.mat \n");
    printf("Measure variable importance in the random forest.\n");
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
    const char *model_file = argv[3];
    const char *save_file = argv[4];
    
    // read frame number, feature, label
    vector<VectorXf> features;
    vector<int> labels;
    readDataset(feature_file, label_file, features, labels);
    const int category_num = *std::max_element(labels.begin(), labels.end()) + 1; //@todo
    printf("category number is %d\n", category_num);
    
    // load model
    DTClassifier model;
    bool is_read = model.load(model_file);
    assert(is_read);
    
    Eigen::VectorXd variable_importantc = model.measureVariableImportance(features, labels);
    
    // save result
    Eigen::MatrixXd result(variable_importantc.size(), 1);
    for (int i = 0; i<variable_importantc.size(); i++) {
        result(i, 0) = variable_importantc[i];
    }
    
    //DTUtil_IO::save_matrix<MatrixXi>(save_file, result);
    matio::writeMatrix<Eigen::MatrixXd>(save_file, "variable_importance", result);
    return 0;
}
#endif

