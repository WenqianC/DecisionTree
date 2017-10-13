//
//  ut_otfi.cpp
//  SequentialRandomForest
//
//  Created by jimmy on 2017-07-17.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "ut_otfi.h"
#include "mat_io.hpp"
#include "otfi_tree.hpp"
#include "dt_util.hpp"
#include <limits>
#include <iostream>
#include "otfi_classifier.hpp"
#include "otfi_classifier_builder.hpp"

using namespace std;


static void read_2_class_synthesitc_dataset(vector<Eigen::VectorXf> & train_features,
                                            vector<int> & train_labels,
                                            vector<Eigen::VectorXf> & test_features,
                                            vector<int> & test_labels)
{
    {
        // train data
        const char file_name1[] = "/Users/jimmy/Desktop/learning_data/synthetic_data/binary_cls/x_train.mat";
        const char file_name2[] = "/Users/jimmy/Desktop/learning_data/synthetic_data/binary_cls/y_train.mat";
        Eigen::MatrixXf features;
        Eigen::MatrixXi labels;
        matio::readMatrix(file_name1, "feature", features);
        matio::readMatrix(file_name2, "label", labels);
        assert(labels.cols() == 1);
        
        for (int i = 0; i < features.rows(); i++) {
            Eigen::VectorXf feat = features.row(i);
            int label = labels(i, 0);
            
            train_features.push_back(feat);
            train_labels.push_back(label);
        }
    }
    assert(train_features.size() == train_labels.size());
    printf("read %lu train examples\n", train_features.size());
    
    {
        // test data
        const char file_name1[] = "/Users/jimmy/Desktop/learning_data/synthetic_data/binary_cls/x_test.mat";
        const char file_name2[] = "/Users/jimmy/Desktop/learning_data/synthetic_data/binary_cls/y_test.mat";
        Eigen::MatrixXf features;
        Eigen::MatrixXi labels;
        matio::readMatrix(file_name1, "feature", features);
        matio::readMatrix(file_name2, "label", labels);
        assert(labels.cols() == 1);
        
        for (int i = 0; i<features.rows(); i++) {
            Eigen::VectorXf feat = features.row(i);
            int label = labels(i, 0);
            
            test_features.push_back(feat);
            test_labels.push_back(label);
        }
    }
    assert(test_features.size() == test_labels.size());
    printf("read %lu test examples\n", test_features.size());
}

static void read_4_class_synthesitc_dataset(vector<Eigen::VectorXf> & train_features,
                                            vector<int> & train_labels,
                                            vector<Eigen::VectorXf> & test_features,
                                            vector<int> & test_labels)
{
    {
        // train data
        const char file_name1[] = "/Users/jimmy/Desktop/learning_data/synthetic_data/multi_cls/x_train.mat";
        const char file_name2[] = "/Users/jimmy/Desktop/learning_data/synthetic_data/multi_cls/y_train.mat";
        Eigen::MatrixXf features;
        Eigen::MatrixXi labels;
        matio::readMatrix(file_name1, "feature", features);
        matio::readMatrix(file_name2, "label", labels);
        assert(labels.cols() == 1);
        
        for (int i = 0; i < features.rows(); i++) {
            Eigen::VectorXf feat = features.row(i);
            int label = labels(i, 0);
            
            train_features.push_back(feat);
            train_labels.push_back(label);
        }
    }
    assert(train_features.size() == train_labels.size());
    printf("read %lu train examples\n", train_features.size());
    
    {
        // test data
        const char file_name1[] = "/Users/jimmy/Desktop/learning_data/synthetic_data/multi_cls/x_test.mat";
        const char file_name2[] = "/Users/jimmy/Desktop/learning_data/synthetic_data/multi_cls/y_test.mat";
        Eigen::MatrixXf features;
        Eigen::MatrixXi labels;
        matio::readMatrix(file_name1, "feature", features);
        matio::readMatrix(file_name2, "label", labels);
        assert(labels.cols() == 1);
        
        for (int i = 0; i<features.rows(); i++) {
            Eigen::VectorXf feat = features.row(i);
            int label = labels(i, 0);
            
            test_features.push_back(feat);
            test_labels.push_back(label);
        }
    }
    assert(test_features.size() == test_labels.size());
    printf("read %lu test examples\n", test_features.size());
}




void test_otfi()
{
    //test_otfi_single_tree();
    test_otfi_multiple_trees();
}

void test_otfi_single_tree()
{
    OTFITree tree;
    vector<Eigen::VectorXf> features;
    vector<int> labels;
    
    vector<Eigen::VectorXf> mdata_features;
    vector<int> mdata_labels;
    read_2_class_synthesitc_dataset(features, labels, mdata_features, mdata_labels);
    

    vector<int> indices = DTUtil::range<int>(0, (int)features.size(), 1);
    vector<int> mdata_indices = DTUtil::range<int>(0, (int)mdata_features.size(), 1);

    // remove some data in the mdata_features
    int max_tree_depth = 10;
    OTFITreeParameter tree_param;
    
    tree_param.category_num_ = 2;
    tree_param.tree_num_ = 2;
    tree_param.max_tree_depth_ = max_tree_depth;
    tree_param.candidate_dim_num_ = 4;
    tree_param.candidate_threshold_num_ = 5;
    tree_param.min_leaf_node_num_ = 5;
    //tree_param.verbose_ = true;
    tree_param.verbose_leaf_ = false;

    tree.buildTree(features, labels, indices, tree_param);

    // add prediction
    vector<int> predictions(mdata_labels.size());
    for (int i =0 ; i<mdata_features.size(); i++) {
        int pred = 0;
        tree.predict(mdata_features[i], pred);
        predictions[i] = pred;
    }
    
    Eigen::MatrixXd conf = DTUtil::confusionMatrix<int>(mdata_labels, predictions, 2, true);
    cout<<"confusion matrix (no missing data): \n"<<conf<<endl<<endl;

    const float missing_mask = std::numeric_limits<float>::max();
    int dims = (int)mdata_features[0].size();
    for (int i = 0;  i<mdata_features.size(); i++){
        int d = rand()%dims;
        mdata_features[i][d] = missing_mask;
    }
    
    for (int i =0 ; i<mdata_features.size(); i++) {
        int pred = 0;
        tree.predict(mdata_features[i], pred);
        predictions[i] = pred;
    }
    
    conf = DTUtil::confusionMatrix<int>(mdata_labels, predictions, 2, true);
    cout<<"confusion matrix (missing data): \n"<<conf<<endl<<endl;

    vector<float> weight(mdata_features.size(), 0);
    tree.imputeFeature(features, labels, indices,
                       mdata_labels, mdata_indices, missing_mask,
                       mdata_features, weight);
    
    for (int i =0 ; i<mdata_features.size(); i++) {
        int pred = 0;
        tree.predict(mdata_features[i], pred);
        predictions[i] = pred;
    }
    
    conf = DTUtil::confusionMatrix<int>(mdata_labels, predictions, 2, true);
    cout<<"confusion matrix (imputed data): \n"<<conf<<endl<<endl;
}

void test_otfi_multiple_trees()
{
    OTFIClassifierBuilder builder;
    int max_tree_depth = 10;
    const int category_num = 2;
    OTFITreeParameter tree_param;

    tree_param.category_num_ = category_num;
    tree_param.tree_num_ = 4;
    tree_param.max_tree_depth_ = max_tree_depth;
    tree_param.candidate_dim_num_ = 4;
    tree_param.candidate_threshold_num_ = 10;
    tree_param.min_leaf_node_num_ = 5;    
    //tree_param.verbose_ = true;
    tree_param.verbose_leaf_ = false;

    
    // repeat 100 times
    Eigen::MatrixXd conf_org = Eigen::MatrixXd::Zero(category_num, category_num);
    Eigen::MatrixXd conf_miss = Eigen::MatrixXd::Zero(category_num, category_num);
    Eigen::MatrixXd conf_impute = Eigen::MatrixXd::Zero(category_num, category_num);
    
    for (int n = 0; n<10; n++) {
        vector<Eigen::VectorXf> features;
        vector<int> labels;
        vector<Eigen::VectorXf> mdata_features;
        vector<int> mdata_labels;
        read_2_class_synthesitc_dataset(features, labels, mdata_features, mdata_labels);
        
        OTFIClassifier model;
        builder.setTreeParameter(tree_param);
        builder.buildModel(model, features, labels, "debug.txt");
        
        // add prediction
        vector<int> predictions(mdata_labels.size());
        for (int i =0 ; i<mdata_features.size(); i++) {
            int pred = 0;
            model.predict(mdata_features[i], pred);
            predictions[i] = pred;
        }
        Eigen::MatrixXd conf = DTUtil::confusionMatrix<int>(mdata_labels, predictions, category_num, true);
        //cout<<"confusion matrix (no missing data): \n"<<conf<<endl<<endl;
        
        conf_org += conf;
        
        const float missing_mask = std::numeric_limits<float>::max();
        int dims = (int)mdata_features[0].size();
        const int missing_dim_num = 5;
        for (int i = 0;  i<mdata_features.size(); i++){
            for (int j = 0; j<missing_dim_num; j++) {
                int d = rand()%dims;
                mdata_features[i][d] = missing_mask;
            }
        }
        
        for (int i =0 ; i<mdata_features.size(); i++) {
            int pred = 0;
            model.predict(mdata_features[i], pred);
            predictions[i] = pred;
        }
        
        conf = DTUtil::confusionMatrix<int>(mdata_labels, predictions, category_num, true);
        //cout<<"confusion matrix (missing data): \n"<<conf<<endl<<endl;
        conf_miss += conf;
        
        
        OTFIClassifier another_model;  // test model read/write
        another_model.load("debug.txt");
        another_model.imputeFeature(features, labels, mdata_features, mdata_labels, missing_mask);
        
        for (int i =0 ; i<mdata_features.size(); i++) {
            int pred = 0;
            model.predict(mdata_features[i], pred);
            predictions[i] = pred;
        }
        
        conf = DTUtil::confusionMatrix<int>(mdata_labels, predictions, category_num, true);
        //cout<<"confusion matrix (imputed data): \n"<<conf<<endl<<endl;
        conf_impute += conf;
    }
    
    conf_org /= 10.0;
    conf_miss /= 10.0;
    conf_impute /= 10.0;
    
    cout<<"confusion matrix (no missing data): \n"<<conf_org<<endl;
    cout<<"accuracy: "<<DTUtil::accuracyFromConfusionMatrix(conf_org).transpose()<<endl<<endl;
    cout<<"confusion matrix (missing data): \n"<<conf_miss<<endl;
    cout<<"accuracy: "<<DTUtil::accuracyFromConfusionMatrix(conf_miss).transpose()<<endl<<endl;
    cout<<"confusion matrix (imputed data): \n"<<conf_impute<<endl;
    cout<<"accuracy: "<<DTUtil::accuracyFromConfusionMatrix(conf_impute).transpose()<<endl<<endl;
}