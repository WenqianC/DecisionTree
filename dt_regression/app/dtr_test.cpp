//
//  dtr_main_imputation.cpp
//  SequentialRandomForest
//
//  Created by jimmy on 2017-11-02.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#if 0
#include <stdio.h>
#include "dt_util_io.hpp"
#include "mat_io.hpp"
#include "dt_util.hpp"
#include "dt_regressor_builder.h"
#include "dt_regressor.h"
#include <iostream>
#include <vector>

using std::cout;
using std::endl;
using std::vector;
using Eigen::VectorXf;

static void help()
{
    printf("program   testXFile  testYFile  modelFilePrefix  resultFile \n");
    printf("DTR_test  test_x.mat test_y.mat model.txt        pred.mat   \n");
    printf("testXFile: testing feature file. 'feature' \n");
    printf("testYFile: testing label file. 'label'  \n");
    printf("resultFile: .mat file, has 'ground_truth' and 'prediction'. \n");
}

static void readRegressionDataset(const char *feature_file,
                                  const char *label_file,
                                  vector<Eigen::VectorXf> & features,
                                  vector<Eigen::VectorXf> & labels)
{
    Eigen::MatrixXf mat_features;
    Eigen::MatrixXf mat_labels;
    matio::readMatrix(feature_file, "feature", mat_features);
    matio::readMatrix(label_file, "label", mat_labels);
    assert(mat_features.rows() == mat_labels.rows());
    
    for (int i = 0; i < mat_features.rows(); i++) {
        Eigen::VectorXf feat = mat_features.row(i);
        Eigen::VectorXf label = mat_labels.row(i);
        
        features.push_back(feat);
        labels.push_back(label);
    }
    assert(features.size() == labels.size());
    printf("read %lu train examples\n", features.size());
}



int main(int argc, const char * argv[])
{
    
    if (argc != 5) {
        printf("argc is %d, should be 5 .\n", argc);
        help();
        return -1;
    }
   
    const char *test_feature_file = argv[1];
    const char *test_label_file = argv[2];
    const char *model_file = argv[3];
    const char *save_file = argv[4];
    
    
    /*
    const char *test_feature_file = "/Users/jimmy/Desktop/cvs_eccv2018_code/data/feature/left_test_fn_feature.mat";
    const char *test_label_file = "/Users/jimmy/Desktop/cvs_eccv2018_code/data/left_test_fn_vq.mat";
    const char *model_file = "/Users/jimmy/Desktop/cvs_eccv2018_code/dtr_train/model/left.txt";
    const char *save_file = "result.mat";
     */
    
    // read data
    vector<Eigen::VectorXf> test_features;
    vector<Eigen::VectorXf> test_labels;
    readRegressionDataset(test_feature_file, test_label_file, test_features, test_labels);
    assert(test_features.size() == test_labels.size());
    const int dims = (int)test_labels[0].size();
    
    // final result
    Eigen::MatrixXf predictions(test_labels.size(), dims);
    Eigen::MatrixXf groundtruth(test_labels.size(), dims);
    
    // record ground truth
    for (int i = 0; i<test_labels.size(); i++) {
        groundtruth.row(i) = test_labels[i];
    }
    
    // treat each dimension as a regression problem;
    DTRegressor model;
    model.load(model_file);
    for (int i = 0; i<test_features.size(); i++) {
        Eigen::VectorXf pred;
        bool is_pred = model.predict(test_features[i], pred);
        assert(is_pred);
        predictions.row(i) = pred;
    }
   
    std::vector<std::string> var_name;
    std::vector<Eigen::MatrixXf> data;
    var_name.push_back("ground_truth");
    var_name.push_back("prediction");
    data.push_back(groundtruth);
    data.push_back(predictions);
    
    // save result
    matio::writeMultipleMatrix<Eigen::MatrixXf>(save_file, var_name, data);
    
    return 0;
}
#endif

