//
//  otfi_imputation.cpp
//  SequentialRandomForest
//
//  Created by jimmy on 2017-10-11.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#if 0

#include <stdio.h>
#include <iostream>

#include "otfi_classifier.hpp"
#include "otfi_classifier_builder.hpp"
#include "dt_util_io.hpp"
#include <iostream>
#include <algorithm>
#include <fstream>
#include <Eigen/Dense>
#include "mat_io.hpp"
#include "dt_util.hpp"

using namespace::std;

static void help()
{
    printf("program          featureFile labelFile modelFile     featureFile_1       labelFile_2  saveFile \n");
    printf("OTFI_imputation  feature.mat label.mat dt_model.txt  missed_feature.mat  label_2.mat  feature_imputed.mat \n");
    printf("Random survival forest, only support real-value feature. Default missing value is float_max\n");
    printf("feature file: .mat file has a 'feature' variable. training feature\n");
    printf("label file:   .mat file has a 'label' variable.   training label\n");
    printf("modelFile: .txt file. The trained model. \n");
    printf("featureFile_1: .mat file has a 'feature' variable. It has missing values\n");
    printf("labelFile_2:   .mat file has a 'label' variable. It has no missing values\n");
    printf("saveFile: .mat file. imputed feautre. \n");
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
    printf("read %lu examples\n", features.size());
}

static bool isSameValue(const float v1, const float v2)
{
    return fabsf(v1 - v2) < 0.00001;
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
    const char *missing_feature_file = argv[4];
    const char *missing_label_file = argv[5];
    const char *save_file = argv[6];
    
    /*
    const char *feature_file = "/Users/jimmy/Desktop/Imputation_RF/main_aux_feature_2.mat";
    const char *label_file = "/Users/jimmy/Desktop/Imputation_RF/main_aux_label_2.mat";
    const char *model_file = "/Users/jimmy/Desktop/Imputation_RF/model/debug.txt";
    const char *missing_feature_file = "/Users/jimmy/Desktop/Imputation_RF/aux_incomplete_feature.mat";
    const char *missing_label_file = "/Users/jimmy/Desktop/Imputation_RF/aux_incomplete_label.mat";
    const char *save_file = "imputed_feature.mat";
     */
    
    // read frame number, feature, label
    vector<VectorXf> features;
    vector<int> labels;
    readDataset(feature_file, label_file, features, labels);
    const int category_num = *std::max_element(labels.begin(), labels.end()) + 1; //@todo
    printf("category number is %d\n", category_num);
    
    vector<Eigen::VectorXf> missing_features;
    vector<int> missing_labels;
    readDataset(missing_feature_file, missing_label_file, missing_features, missing_labels);
    
    // check missing percentage
    const float missing_mask = std::numeric_limits<float>::max();
    int missing_num = 0;
    for (int i = 0; i<missing_features.size(); i++) {
        for (int j = 0; j<missing_features[j].size(); j++) {
            if (isSameValue(missing_features[i][j], missing_mask)) {
                missing_num++;
            }
        }
    }
    printf("missing value percentage is %f\n", 1.0 * missing_num/(missing_features.size() * missing_features[0].size()));
    
    // load model
    OTFIClassifier model;
    bool is_read = model.load(model_file);
    assert(is_read);
    
    // test before imputation
    {
        vector<int> predictions;
        for (int i =0 ; i<missing_features.size(); i++) {
            int pred = 0;
            model.predict(missing_features[i], pred);
            predictions.push_back(pred);
        }
        
        Eigen::MatrixXd conf = DTUtil::confusionMatrix<int>(predictions, missing_labels, category_num, true);
        cout<<"confusion matrix (before imputation): \n"<<conf<<endl;
        cout<<"accuray: "<<DTUtil::precisionFromConfusionMatrix(conf).transpose()<<endl<<endl;
    }
    
    model.imputeFeature(features, labels, missing_features, missing_labels, missing_mask);
    
    // test after imputation
    Eigen::MatrixXd prediction_mask((int)missing_features.size(), 1);
    {
        vector<int> predictions;
        for (int i =0 ; i<missing_features.size(); i++) {
            int pred = 0;
            model.predict(missing_features[i], pred);
            predictions.push_back(pred);
            if (pred == missing_labels[i]) {
                prediction_mask(i, 0) = 1;
            }
            else {
                prediction_mask(i, 0) = 0;
            }
        }
        
        Eigen::MatrixXd conf = DTUtil::confusionMatrix<int>(predictions, missing_labels, category_num, true);
        cout<<"confusion matrix (after imputation): \n"<<conf<<endl;
        cout<<"accuray: "<<DTUtil::precisionFromConfusionMatrix(conf).transpose()<<endl<<endl;
    }
    
    // save imputed feature
    int n = (int)missing_features.size();
    int dim = (int)missing_features[0].size();
    Eigen::MatrixXd imputed_feature(n, dim);
    for (int i = 0; i<n; i++) {
        for (int j = 0; j<dim; j++) {
            imputed_feature(i, j) = missing_features[i][j];
        }
    }
  
    vector<string> var_name;
    vector<Eigen::MatrixXd> var_data;
    var_name.push_back("feature");
    var_name.push_back("prediction_mask");
    var_data.push_back(imputed_feature);
    var_data.push_back(prediction_mask);
    matio::writeMultipleMatrix<Eigen::MatrixXd>(save_file, var_name, var_data);
    
    return 0;
}
#endif
