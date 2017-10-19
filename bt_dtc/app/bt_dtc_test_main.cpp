//
//  bt_dtc_test.cpp
//  PTZBTRF
//
//  Created by jimmy on 2017-10-14.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#if 0
#include <stdio.h>
#include <iostream>
#include "bt_dt_classifier.h"
#include "bt_dt_classifier_builder.h"
#include "dt_util_io.hpp"
#include <algorithm>
#include <fstream>
#include <Eigen/Dense>
#include "mat_io.hpp"
#include "dt_util.hpp"

using namespace::std;


static void help()
{
    printf("program     featureFile labelFile modelFile  max_check dist_threshold saveFile \n");
    printf("BTDTC_test  feature.mat label.mat model.txt  2         -1             gd_pred.mat \n");
    printf("feature file: .mat file has a 'feature' variable. \n");
    printf("label file:   .mat file has a 'label' variable. \n");
    printf("Save inlier predictions. Outlier prediction is saved as -1 \n");
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
    if (argc != 7) {
        printf("argc is %d, should be 7.\n", argc);
        help();
        return -1;
    }
    
    const char *feature_file = argv[1];
    const char *label_file = argv[2];
    const char *model_file = argv[3];
    const int max_check = (int)strtod(argv[4], NULL);
    float dist_threshold = (float)strtod(argv[5], NULL);
    const char *save_file = argv[6];
    
    if (dist_threshold < 0) {
        dist_threshold = INT_MAX;
    }
    // read frame number, feature, label
    vector<VectorXf> features;
    vector<int> labels;
    readDataset(feature_file, label_file, features, labels);
    
    // load model
    BTDTClassifier model;
    bool is_read = model.load(model_file);
    assert(is_read);
    const int category_num = model.categoryNum();
    printf("category number is %d\n", category_num);

    
    // predict
    vector<int> inlier_predictions;
    vector<int> inlier_labels;
    vector<int> all_predictions;
    int outlier_num = 0;
    for (int i = 0; i<features.size(); i++) {
        int pred = 0;
        bool is_pred = model.predict(features[i], max_check, pred, dist_threshold);
        if (is_pred) {
            inlier_predictions.push_back(pred);
            inlier_labels.push_back(labels[i]);
        }
        else {
            outlier_num++;
            pred = -1;
        }
        all_predictions.push_back(pred);
    }
    assert(inlier_predictions.size() == inlier_labels.size());
    Eigen::MatrixXd confusion = DTUtil::confusionMatrix<int>(inlier_predictions, inlier_labels, category_num, false);
    Eigen::VectorXd accuracy = DTUtil::accuracyFromConfusionMatrix(confusion);
    cout<<"Test confusion matrix: \n"<<confusion<<endl;
    cout<<"Test precision: \n"<<accuracy.transpose()<<endl;
    printf("Outlier percentage %lf\n", 1.0 * outlier_num/features.size());
    
    assert(all_predictions.size() == labels.size());
    
    // only prediction result
    Eigen::MatrixXd result((int)all_predictions.size(), 2);
    for (int i = 0; i<labels.size(); i++) {
        result(i, 0) = labels[i];
        result(i, 1) = (int)all_predictions[i];
    }
    
    matio::writeMatrix<Eigen::MatrixXd>(save_file, "ground_truth_prediction", result);
    return 0;
}
#endif
