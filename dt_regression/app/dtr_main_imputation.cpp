//
//  dtr_main_imputation.cpp
//  SequentialRandomForest
//
//  Created by jimmy on 2017-11-02.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#if 1
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
    printf("program      trainXFile  trainYFile  testXFile  testYFile  useTestY DTParamFile   modelFilePrefix  resultFile \n");
    printf("DTR_impute   train_x.mat train_y.mat test_x.mat test_y.mat 1        dtr_param.txt result/dtc_model imputed_feature.mat \n");
    printf("trainXFile: training feature file. .mat 'feature' \n");
    printf("trainYFile: training label file.   .mat 'label' \n");
    printf("testXFile: testing feature file.  \n");
    printf("testYFile: testing label file. optional  \n");
    printf("useTestY: 0 --> no testYFile, 1 --> has testYFile \n");
    printf("DTParamFile: regression tree parameter \n");
    printf("modelFilePrefix: profix of trained model file.\n");
    printf("resultFile: .mat file, has 'imputed_feature'. \n");
}

static vector<Eigen::VectorXf> getOneDimension(const vector<Eigen::VectorXf> & data, const int dim)
{
    assert(dim >= 0 && dim < data[0].size());
    
    vector<Eigen::VectorXf> single_dim_data;
    VectorXf temp_data(1);
    for (int i = 0; i<data.size(); i++) {
        temp_data[0] = data[i][dim];
        single_dim_data.push_back(temp_data);
    }
    return single_dim_data;
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

static void readRegressionFeature(const char *feature_file,
                                  vector<Eigen::VectorXf> & features)
{
    Eigen::MatrixXf mat_features;
    Eigen::MatrixXf mat_labels;
    matio::readMatrix(feature_file, "feature", mat_features);
    
    for (int i = 0; i < mat_features.rows(); i++) {
        Eigen::VectorXf feat = mat_features.row(i);
        features.push_back(feat);
        
    }
    printf("read %lu feature examples\n", features.size());
}



int main(int argc, const char * argv[])
{
    if (argc != 9) {
        printf("argc is %d, should be 9 .\n", argc);
        help();
        return -1;
    }
    
    const char *train_feature_file = argv[1];
    const char *train_label_file   = argv[2];
    const char *test_feature_file = argv[3];
    const char *test_label_file = argv[4];
    const int has_testing_label = (int)strtod(argv[5], NULL);
    const char *param_file = argv[6];
    const char *model_file_prefix = argv[7];
    const char *save_file = argv[8];
    
    assert(has_testing_label == 1 || has_testing_label == 0);
    
    /*
    const char *train_feature_file = "/Users/jimmy/Desktop/dtr_imputation/data/train_x.mat";
    const char *train_label_file   = "/Users/jimmy/Desktop/dtr_imputation/data/train_y.mat";
    const char *test_feature_file = "/Users/jimmy/Desktop/dtr_imputation/data/test_x.mat";
    const char *test_label_file = "/Users/jimmy/Desktop/dtr_imputation/data/test_y.mat";
    const char *param_file = "/Users/jimmy/Desktop/dtr_imputation/dtr_tree_param.txt";
    const char *model_file = "debug.txt";
    const char *save_file = "result.mat";
     */
    
    vector<Eigen::VectorXf> train_features;
    vector<Eigen::VectorXf> train_labels;
    readRegressionDataset(train_feature_file, train_label_file, train_features, train_labels);
    
    vector<Eigen::VectorXf> test_features;
    vector<Eigen::VectorXf> test_labels;
    if (has_testing_label) {
        readRegressionDataset(test_feature_file, test_label_file, test_features, test_labels);
    }
    else {
        readRegressionFeature(test_feature_file, test_features);
    }
    assert(test_features.size() != 0);
    
    const int dims = (int)train_labels[0].size();
    printf("imputing %d dimeion features \n", dims);
    
    // final result
    Eigen::MatrixXf predictions(test_features.size(), dims);
    
    DTRTreeParameter param;
    param.readFromFile(param_file);
    param.feature_dimension_ = (int)train_features[0].size();
    cout<<"decition tree parameter "<<param<<endl;
    
    // treat each dimension as a regression problem;
    for (int d = 0; d<dims; d++) {
        char model_file[1024] = {NULL};
        sprintf(model_file, "%s_%d.txt", model_file_prefix, d);
        // get training label
        vector<Eigen::VectorXf> cur_train_labels = getOneDimension(train_labels, d);
        
        DTRegressor model;
        DTRegressorBuilder builder;
        builder.setTreeParameter(param);
        
        if (has_testing_label) {
            vector<Eigen::VectorXf> cur_test_labels  = getOneDimension(test_labels, d);
            builder.buildModel(model, train_features, cur_train_labels,
                               test_features,  cur_test_labels, model_file);
        }
        else {
            builder.buildModel(model, train_features, cur_train_labels,
                               vector<Eigen::VectorXf>(),  vector<Eigen::VectorXf>(), model_file);
        }
        
        // predict on current dimension
        for (int i = 0; i<test_features.size(); i++) {
            Eigen::VectorXf pred;
            bool is_pred = model.predict(test_features[i], pred);
            assert(is_pred);
            assert(pred.size() == 1);
            predictions(i, d) = pred[0];
        }
    }
    // save result
    matio::writeMatrix<Eigen::MatrixXf>(save_file, "imputed_feature", predictions);
    
    return 0;
}
#endif

