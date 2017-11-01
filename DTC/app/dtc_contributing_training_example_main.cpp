//
//  main.cpp
//  Classifer_RF
//
//  Created by jimmy on 2016-07-26.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#if 1
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

// feature.mat label.mat

static void help()
{
    printf("program                    trainFeatureFile  modelFile    testFeatureFile  testLabelFile  mostCommon saveFile \n");
    printf("DTC_contributing_analysis  train_feature.mat dt_model.txt test_feature.mat test_label.mat 0          contributing_index.mat \n");
    printf("Record contributing training example in the random forest.\n");
    printf("Node: out-of-bag example is simulated, not exactly the same as in training, but statistically be very similar.\n");
    printf("trainFeatureFile: .mat file has a 'feature' variable. \n");
    printf("testFeatureFile file: .mat file has a 'feature' variable. \n");
    printf("testLabelFile file:   .mat file has a 'label' variable. \n");
    printf("mostCommon: 1 --> only save the most common index in leaf node. \n");
    printf("saveFile: .mat file has 'ground_truth_prediction' and 'contributing_train_index', (-1) for empty cell\n");
}

static void readFeature(const char *feature_file,
                        vector<Eigen::VectorXf> & features)
{
    Eigen::MatrixXf mat_features;
    matio::readMatrix(feature_file, "feature", mat_features);
    
    for (int i = 0; i < mat_features.rows(); i++) {
        Eigen::VectorXf feat = mat_features.row(i);
        features.push_back(feat);
    }
    printf("read %lu features\n", features.size());
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

int main(int argc, const char * argv[])
{
    if (argc != 7) {
        printf("argc is %d, should be 7.\n", argc);
        help();
        return -1;
    }
    
    const char *train_feature_file = argv[1];
    const char *model_file = argv[2];
    
    const char *test_feature_file = argv[3];
    const char *test_label_file = argv[4];
    const bool most_common = ((int)strtod(argv[5], NULL) != 0);
    const char *save_file = argv[6];
    
    // read training feature
    vector<VectorXf> train_features;
    readFeature(train_feature_file, train_features);
    
    // read testing feaure and label
    vector<VectorXf> test_features;
    vector<int> test_labels;
    readDataset(test_feature_file, test_label_file, test_features, test_labels);
    const int category_num = *std::max_element(test_labels.begin(), test_labels.end()) + 1; //@todo
    
   
    printf("category number is %d\n", category_num);
    
    // load model
    DTClassifier model;
    bool is_read = model.load(model_file);
    assert(is_read);
    
    const bool simulate_out_of_bag_sampling = true;
    vector<int> predictions;
    vector< vector<int> > cti; // contribute training index
    int max_num_index = 0;
    for (int i = 0; i<test_features.size(); i++) {
        int pred = 0;
        vector<int> cur_index;
        model.getContributingExamples(train_features, test_features[i], simulate_out_of_bag_sampling, pred, cur_index);
        predictions.push_back(pred);
        cti.push_back(cur_index);
        if (cur_index.size() > max_num_index) {
            max_num_index = (int)cur_index.size();
        }
    }
    printf("maximum number of contributing example is %d.\n", max_num_index);
    Eigen::MatrixXd confusion = DTUtil::confusionMatrix<int>(predictions, test_labels, category_num, false);
    Eigen::VectorXd accuracy = DTUtil::accuracyFromConfusionMatrix(confusion);
    cout<<"Test confusion matrix: \n"<<confusion<<endl;
    cout<<"Test accuracy: \n"<<accuracy.transpose()<<endl;
    
    assert(predictions.size() == test_labels.size());
    assert(predictions.size() == cti.size());
    
    if (most_common) {
        // only keep the most common index
        for (int i = 0; i<cti.size(); i++) {
            if (cti[i].size() > 0) {
                int most_index = dt::mostCommon(cti[i]);
                cti[i].resize(1);
                cti[i][0] = most_index;
            }
            else {
                cti[i].resize(1);
                cti[i][0] = -1;
            }
        }
        max_num_index = 1;
    }
    
    // save result
    Eigen::MatrixXd result((int)predictions.size(), 2);
    for (int i = 0; i<test_labels.size(); i++) {
        result(i, 0) = test_labels[i];
        result(i, 1) = (int)predictions[i];
    }
    
    Eigen::MatrixXd contributing_index_mat((int)cti.size(), max_num_index);
    contributing_index_mat.fill(-1);
    for (int i = 0; i<cti.size(); i++) {
        for (int j = 0; j<cti[i].size(); j++) {
            contributing_index_mat(i, j) = cti[i][j];
        }
    }
    
   
    vector<string> var_name;
    vector<Eigen::MatrixXd> var_data;
    var_name.push_back("ground_truth_prediction");
    var_name.push_back("contributing_train_index");
    var_data.push_back(result);
    var_data.push_back(contributing_index_mat);
    matio::writeMultipleMatrix(save_file, var_name, var_data);
    
    return 0;
}
#endif

