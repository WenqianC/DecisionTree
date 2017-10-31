//
//  UT_decision_tree_classifier.cpp
//  Classifer_RF
//
//  Created by jimmy on 2016-07-27.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#include "ut_dtc.h"
#include "dtc_tree.h"
#include "dt_util.hpp"
#include <iostream>
#include "dt_classifier.h"
#include "dt_classifier_builder.h"
#include "dt_util_io.hpp"
#include "dt_param_parser.h"
#include "mat_io.hpp"


using Eigen::MatrixXd;
using std::cout;
using std::endl;

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


void test_DTC_synthesized_data()
{
 //  test_synthsized_data_single_tree();
 //   test_synthesized_data_multiple_trees();
//    test_parameter_parser();
//    test_synthesized_data_contribute_training_example_single_tree();
    test_synthesized_data_contribute_training_example_multiple_tree();
}

void test_synthsized_data_single_tree()
{
    vector<VectorXf> features;
    vector<int> labels;
    
    DTCTreeParameter param(10, 3);
    param.max_depth_ = 5;
    param.min_leaf_node_num_ = 5;
    param.min_split_num_ = 8;
    param.split_candidate_num_ = 10;
    param.category_num_ = 4;
    param.balanced_example_ = true;
    param.verbose_ = false;
   
    const char train_feature_file[] = "/Users/jimmy/Desktop/learning_data/synthetic_data/multi_cls/x_train.mat";
    const char train_label_file[]   = "/Users/jimmy/Desktop/learning_data/synthetic_data/multi_cls/y_train.mat";
    
    readDataset(train_feature_file, train_label_file, features, labels);
    
    
    vector<int> indices;
    for (int i =0; i<labels.size(); i++) {
        indices.push_back(i);
    }
    DTCTree tree;
    tree.buildTree(features, labels, indices, param);
    
    // test on test data
    const char test_feature_file[] = "/Users/jimmy/Desktop/learning_data/synthetic_data/multi_cls/x_test.mat";
    const char test_label_file[]   = "/Users/jimmy/Desktop/learning_data/synthetic_data/multi_cls/y_test.mat";
    vector<VectorXf> test_features;
    vector<int> test_labels;
    readDataset(test_feature_file, test_label_file, test_features, test_labels);
    
    int correct_num = 0;
    Eigen::MatrixXd confusion = Eigen::MatrixXd::Zero(param.category_num_, param.category_num_);
    for (int i = 0; i<test_features.size(); i++) {
        Eigen::VectorXf prob;
        int pred = 0;
        int gd = test_labels[i];
        tree.predict(test_features[i], prob);
        prob.maxCoeff(& pred);
        if (pred == gd) {
            correct_num++;
        }
        confusion(gd, pred) += 1.0;
    }
    printf("precision is %f\n", 1.0 * correct_num/test_labels.size());
    cout<<"confusion matrix is \n"<<confusion<<endl;
}


void test_synthesized_data_multiple_trees()
{
    vector<VectorXf> features;
    vector<int> labels;
    
    DTCTreeParameter param(10, 3);
    param.max_depth_ = 5;
    param.min_leaf_node_num_ = 5;
    param.min_split_num_ = 8;
    param.split_candidate_num_ = 10;
    param.category_num_ = 4;
    param.balanced_example_ = true;
    param.verbose_ = false;
    
    const char train_feature_file[] = "/Users/jimmy/Desktop/learning_data/synthetic_data/multi_cls/x_train.mat";
    const char train_label_file[]   = "/Users/jimmy/Desktop/learning_data/synthetic_data/multi_cls/y_train.mat";
    
    readDataset(train_feature_file, train_label_file, features, labels);
    
    // test on test data
    const char test_feature_file[] = "/Users/jimmy/Desktop/learning_data/synthetic_data/multi_cls/x_test.mat";
    const char test_label_file[]   = "/Users/jimmy/Desktop/learning_data/synthetic_data/multi_cls/y_test.mat";
    vector<VectorXf> test_features;
    vector<int> test_labels;
    readDataset(test_feature_file, test_label_file, test_features, test_labels);
    
    // train the model
    DTClassifierBuilder builder;
    DTClassifier model;
    builder.setTreeParameter(param);
    builder.buildModel(model, features, labels, test_features, test_labels);
    
    model.save("dtc_model.txt");
    
    
    DTClassifier model2;
    model2.load("dtc_model.txt");
    
    
    // test on test data
    
    vector<int> predictions;
    for (int i = 0; i<test_features.size(); i++) {
        int pred;
        model2.predict(test_features[i], pred);
        predictions.push_back(pred);
    }
    
    Eigen::MatrixXd confusion = DTUtil::confusionMatrix(predictions, test_labels, param.category_num_, true);
    
    cout<<"test confusion matrix is \n"<<confusion<<endl;
}

void test_synthesized_data_contribute_training_example_single_tree()
{
    vector<VectorXf> features;
    vector<int> labels;
    
    DTCTreeParameter param(10, 3);
    param.max_depth_ = 20;
    param.min_leaf_node_num_ = 10;
    param.min_split_num_ = 8;
    param.split_candidate_num_ = 10;
    param.category_num_ = 4;
    param.balanced_example_ = true;
    param.verbose_ = false;
    
    const char train_feature_file[] = "/Users/jimmy/Desktop/learning_data/synthetic_data/multi_cls/x_train.mat";
    const char train_label_file[]   = "/Users/jimmy/Desktop/learning_data/synthetic_data/multi_cls/y_train.mat";
    
    readDataset(train_feature_file, train_label_file, features, labels);
    
    
    vector<int> indices;
    for (int i =0; i<labels.size(); i++) {
        indices.push_back(i);
    }
    DTCTree tree;
    tree.buildTree(features, labels, indices, param);
    
    // test on test data
    const char test_feature_file[] = "/Users/jimmy/Desktop/learning_data/synthetic_data/multi_cls/x_test.mat";
    const char test_label_file[]   = "/Users/jimmy/Desktop/learning_data/synthetic_data/multi_cls/y_test.mat";
    vector<VectorXf> test_features;
    vector<int> test_labels;
    readDataset(test_feature_file, test_label_file, test_features, test_labels);
    
    
    int correct_num = 0;
    Eigen::MatrixXd confusion = Eigen::MatrixXd::Zero(param.category_num_, param.category_num_);
    for (int i = 0; i<test_features.size(); i++) {
        Eigen::VectorXf prob;
        int pred = 0;
        int gd = test_labels[i];
        vector<int> contribute_training_index;
        tree.analyzePrediction(features, indices, test_features[i], prob, contribute_training_index);
        prob.maxCoeff(&pred);
        if (pred == gd) {
            correct_num++;
        }
        confusion(gd, pred) += 1.0;
        
        // print contribute training example index
        printf("contribute training example index: ");
        for (int j = 0; j<contribute_training_index.size(); j++) {
            printf("%d ", contribute_training_index[j]);
        }
        printf("\n");
    }
    printf("precision is %f\n", 1.0 * correct_num/test_labels.size());
     
    cout<<"confusion matrix is \n"<<confusion<<endl;    
}

void test_synthesized_data_contribute_training_example_multiple_tree()
{
    vector<VectorXf> features;
    vector<int> labels;
    
    DTCTreeParameter param(10, 3);
    param.max_depth_ = 5;
    param.min_leaf_node_num_ = 5;
    param.min_split_num_ = 8;
    param.split_candidate_num_ = 10;
    param.category_num_ = 4;
    param.balanced_example_ = true;
    param.verbose_ = false;
    
    const char train_feature_file[] = "/Users/jimmy/Desktop/learning_data/synthetic_data/multi_cls/x_train.mat";
    const char train_label_file[]   = "/Users/jimmy/Desktop/learning_data/synthetic_data/multi_cls/y_train.mat";
    
    readDataset(train_feature_file, train_label_file, features, labels);
    
    // test on test data
    const char test_feature_file[] = "/Users/jimmy/Desktop/learning_data/synthetic_data/multi_cls/x_test.mat";
    const char test_label_file[]   = "/Users/jimmy/Desktop/learning_data/synthetic_data/multi_cls/y_test.mat";
    vector<VectorXf> test_features;
    vector<int> test_labels;
    readDataset(test_feature_file, test_label_file, test_features, test_labels);
    
    // train the model
    DTClassifierBuilder builder;
    DTClassifier model;
    builder.setTreeParameter(param);
    builder.buildModel(model, features, labels, test_features, test_labels);
    
    model.save("dtc_model.txt");
    
    
    DTClassifier model2;
    model2.load("dtc_model.txt");
    
    
    // test on test data
    
    /*
    bool getContributingExamples(const vector<Eigen::VectorXf> & training_features,
                                 const Eigen::VectorXf & valid_feature,
                                 const bool simulate_oob_sampling,
                                 int & pred,
                                 vector<int> & contributing_example_index);
     */
    

    vector<int> predictions;
    bool simulate_oob_sampling = true;
    for (int i = 0; i<test_features.size(); i++) {
        int pred;
        vector<int> contribute_training_index;
        model2.getContributingExamples(features, test_features[i], simulate_oob_sampling, pred, contribute_training_index);
        predictions.push_back(pred);
        
        // print contribute training example index
        printf("contribute training example index: ");
        for (int j = 0; j<contribute_training_index.size(); j++) {
            printf("%d ", contribute_training_index[j]);
        }
        printf("\n");
    }
    
    Eigen::MatrixXd confusion = DTUtil::confusionMatrix(predictions, test_labels, param.category_num_, true);
    
    cout<<"test confusion matrix is \n"<<confusion<<endl;    
}

/*
void test_parameter_parser()
{
    ParameterParser parser;
    parser.loadParameter("/Users/jimmy/Desktop/param_example.txt");
    
    parser.printSelf();
    
    int tree_depth = 0;
    double alpha = 0;
    
    parser.getIntValue("tree_depth", tree_depth);
    parser.getFloatValue("alpha", alpha);
 }
 */


