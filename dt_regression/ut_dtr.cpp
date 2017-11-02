//
//  ut_dtr.cpp
//  SequentialRandomForest
//
//  Created by jimmy on 2017-11-01.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "ut_dtr.h"
#include "dtr_tree.h"
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include "mat_io.hpp"
#include "dt_util.hpp"
#include "dt_regressor_builder.h"
#include "dt_regressor.h"


using namespace::std;
using namespace Eigen;

static void readRegressionDataset(const char *feature_labe_file,
                                  vector<Eigen::VectorXf> & features,
                                  vector<Eigen::VectorXf> & labels)
{
    Eigen::MatrixXf mat_features;
    Eigen::MatrixXf mat_labels;
    matio::readMatrix(feature_labe_file, "feature", mat_features);
    matio::readMatrix(feature_labe_file, "label", mat_labels);
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


void test_dtr_synthesized_data()
{
    //test_dtr_single_tree();
    test_dtr_multiple_trees();
}

void test_dtr_single_tree()
{
    vector<VectorXf> features;
    vector<VectorXf> labels;
    
    DTRTreeParameter param;
    param.max_depth_ = 10;
    param.min_leaf_node_num_ = 5;
    param.min_split_num_ = 8;
    param.split_candidate_num_ = 10;
    param.feature_dimension_ = 2;
    param.verbose_ = false;
    param.verbose_leaf_ = true;
    
    const char train_feature_label_file[] = "/Users/jimmy/Desktop/learning_data/synthetic_data/2d_regression/train_xy.mat";
    readRegressionDataset(train_feature_label_file, features, labels);
    vector<int> indices = dt::range<int>(0, (int)labels.size(), 1);
    DTRTree tree;
    tree.buildTree(features, labels, indices, param);
    
    tree.writeTree("regression_debug.txt");

    DTRTree tree2;
    tree2.setTreeParameter(param);
    tree2.readTree("regression_debug.txt");
    
    // test
    const char test_feature_label_file[] = "/Users/jimmy/Desktop/learning_data/synthetic_data/2d_regression/test_xy.mat";
    vector<VectorXf> test_features;
    vector<VectorXf> test_labels;
    readRegressionDataset(test_feature_label_file, test_features, test_labels);
    
    vector<VectorXf> errors;
    for (int i = 0; i<test_features.size(); i++) {
        Eigen::VectorXf pred;
        bool is_pred = tree2.predict(test_features[i], pred);
        if (is_pred) {
            Eigen::VectorXf dif = pred - test_labels[i];
            errors.push_back(dif);
        }        
    }
    
    Eigen::VectorXf mean_error;
    Eigen::VectorXf median_error;
    dt::meanMedianError(errors, mean_error, median_error);
    
    cout<<"mean error "<<mean_error<<" median error "<<median_error<<endl;
}

void test_dtr_multiple_trees()
{
    vector<VectorXf> features;
    vector<VectorXf> labels;
    
    DTRTreeParameter param;
    param.tree_num_ = 3;
    param.max_depth_ = 10;
    param.min_leaf_node_num_ = 5;
    param.min_split_num_ = 8;
    param.split_candidate_num_ = 10;
    param.feature_dimension_ = 2;
    param.verbose_ = false;
    param.verbose_leaf_ = false;
    
    // train
    const char train_feature_label_file[] = "/Users/jimmy/Desktop/learning_data/synthetic_data/2d_regression/train_xy.mat";
    readRegressionDataset(train_feature_label_file, features, labels);
    
    // test
    const char test_feature_label_file[] = "/Users/jimmy/Desktop/learning_data/synthetic_data/2d_regression/test_xy.mat";
    vector<VectorXf> test_features;
    vector<VectorXf> test_labels;
    readRegressionDataset(test_feature_label_file, test_features, test_labels);
    
    DTRegressor model;
    DTRegressorBuilder builder;
    builder.setTreeParameter(param);
    builder.buildModel(model, features, labels, test_features, test_labels, "debug.txt");
    
    DTRegressor model2;
    model2.load("debug.txt");
    
    
    vector<VectorXf> errors;
    for (int i = 0; i<test_features.size(); i++) {
        Eigen::VectorXf pred;
        bool is_pred = model2.predict(test_features[i], pred);
        if (is_pred) {
            Eigen::VectorXf dif = pred - test_labels[i];
            errors.push_back(dif);
        }
    }
    
    Eigen::VectorXf mean_error;
    Eigen::VectorXf median_error;
    dt::meanMedianError(errors, mean_error, median_error);
    
    cout<<"mean error "<<mean_error<<" median error "<<median_error<<endl;

    
}





