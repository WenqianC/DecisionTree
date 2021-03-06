//
//  DTClassifierBuilder.cpp
//  Classifer_RF
//
//  Created by jimmy on 2016-09-20.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#include "dt_classifier_builder.h"
#include "dt_random.hpp"
#include "dt_util.hpp"

#include <iostream>
#include <thread>
#include <mutex>

using std::cout;
using std::endl;

void DTClassifierBuilder::setTreeParameter(const DTCTreeParameter & param)
{
    tree_param_ = param;
}

bool DTClassifierBuilder::buildModel(DTClassifier & model,
                                     const vector<VectorXf> & features,
                                     const vector<int> & labels,
                                     const vector<VectorXf> & valid_features,
                                     const vector<int>& valid_labels,
                                     const char * model_file_name) const
{
    assert(features.size() == labels.size());
    assert(valid_features.size() == valid_labels.size());
    
    model.tree_param_ = tree_param_;
    model.trees_.clear();
    const int tree_num = tree_param_.tree_num_;
    const int category_num = tree_param_.category_num_;
    const bool is_balance = tree_param_.balanced_example_;
    const int N = (int)features.size();
    
    DTRandom rng;
    for (int n = 0; n<tree_num; n++) {        
        // bagging
        vector<int> training_indices;
        vector<int> validation_indices;
        rng.outofBagSample<int>(N, training_indices, validation_indices);
                
        if (is_balance) {
            vector<int> balanced_indices = dt::balanceSamples<int>(training_indices, labels, category_num);
            printf("balanced example vs unbalanced example ratio: %lf\n", 1.0*balanced_indices.size()/training_indices.size());
            training_indices = balanced_indices;
        }
        DTCTree * tree = new DTCTree();
        assert(tree);
        double tt = clock();
        tree->buildTree(features, labels, training_indices, tree_param_);
        model.trees_.push_back(tree);
        
        printf("build tree %d cost %lf minutes\n", n, (clock()- tt)/CLOCKS_PER_SEC/60.0);
        
        // test on the validation data
        vector<int> cv_predictions;
        vector<int> cv_labels;
        for (int i = 0; i<validation_indices.size(); i++) {
            const int index = validation_indices[i];
            int pred = 0;;
            model.predict(features[index], pred);
            cv_predictions.push_back(pred);
            cv_labels.push_back(labels[index]);            
        }
        
        Eigen::MatrixXd oob_conf = DTUtil::confusionMatrix(cv_predictions, cv_labels, category_num, false);
        cout<<"Out of bag validation confusion matrix: \n"<<oob_conf<<endl;
        if (model_file_name != NULL) {
            model.save(model_file_name);
        }
        
        if (valid_features.size() != 0) {
            vector<int> valid_predictions;
            for (int i = 0; i<valid_features.size(); i++) {
                int pred = 0;
                model.predict(valid_features[i], pred);
                valid_predictions.push_back(pred);
            }
            assert(valid_predictions.size() == valid_labels.size());
            Eigen::MatrixXd valid_conf = DTUtil::confusionMatrix<int>(valid_predictions, valid_labels, category_num, false);
            Eigen::VectorXd accuracy = DTUtil::accuracyFromConfusionMatrix(valid_conf);
            cout<<"Validation confusion matrix: \n"<<valid_conf<<endl;
            cout<<"Validation accuracy: \n"<<accuracy.transpose()<<endl;
        }
    }
    return true;
}


void trainTreeHelper(DTCTree* tree,
                     const vector<VectorXf> & features,
                     const vector<int> & labels,
                     const vector<int> & indices,
                     const DTCTreeParameter & param)
{
    tree->buildTree(features, labels, indices, param);
}

bool DTClassifierBuilder::buildModel(DTClassifier & model,
                                     const vector<VectorXf> & features,
                                     const vector<int> & labels,
                                     const vector<VectorXf> & valid_features,
                                     const vector<int>& valid_labels,
                                     const int thread_num,
                                     const char * model_file_name) const
{
    assert(features.size() == labels.size());
    assert(valid_features.size() == valid_labels.size());
    assert(thread_num >= 1 && thread_num <= 8);
    
    model.tree_param_ = tree_param_;
    model.trees_.clear();
    const int tree_num = tree_param_.tree_num_;
    const int category_num = tree_param_.category_num_;
    const bool is_balance = tree_param_.balanced_example_;
    const int N = (int)features.size();
    
    
    DTRandom rng;
    vector<vector<int>> training_indices(tree_num);
    vector<vector<int>> validation_indices(tree_num);
    for (int n = 0; n<tree_num; n++) {
        // bagging
        rng.outofBagSample<int>(N, training_indices[n], validation_indices[n]);
        
        // balance labels
        if (is_balance) {
            vector<int> balanced_indices = dt::balanceSamples<int>(training_indices[n], labels, category_num);
            printf("balanced example vs unbalanced example ratio: %lf\n", 1.0*balanced_indices.size()/training_indices.size());
            training_indices[n] = balanced_indices;
        }
        DTCTree * tree = new DTCTree();
        assert(tree);
        model.trees_.push_back(tree);
    }
    
    int block_num = tree_num/thread_num;
    int res_num = tree_num - block_num * thread_num;
    
    time_t train_start;
    time_t train_end;
    time(&train_start);
    // build tree in each block
    for (int i = 0; i<block_num; i++) {
        std::vector<std::thread> th;
        for (int j = 0; j<thread_num; j++) {
            int k = i * thread_num + j;
            th.push_back(std::thread(trainTreeHelper, model.trees_[k], features, labels, training_indices[k], tree_param_));
        }
        for (auto &t: th) {
            t.join();
        }
    }
    
    // build the rest trees
    std::vector<std::thread> th;
    for (int i = 0; i<res_num; i++) {
        int k = block_num * thread_num + i;
        th.push_back(std::thread(trainTreeHelper, model.trees_[k], features, labels, training_indices[k], tree_param_));
    }
    for (auto &t: th) {
        t.join();
    }
    time(&train_end);
    
    printf("thread number is %d.\n", thread_num);
    std::cout<<"training takes: "<<difftime(train_end, train_start) << " seconds." << std::endl;
    
    // save and validation
    if (model_file_name != NULL) {
        model.save(model_file_name);
    }
    
    for (int n = 0; n < tree_num; n++) {
         // test on the validation data
         vector<int> cv_predictions;
         vector<int> cv_labels;
         for (int i = 0; i<validation_indices[n].size(); i++) {
             const int index = validation_indices[n][i];
             int pred = 0;;
             model.predict(features[index], pred);
             cv_predictions.push_back(pred);
             cv_labels.push_back(labels[index]);
         }
         
         Eigen::MatrixXd oob_conf = DTUtil::confusionMatrix(cv_predictions, cv_labels, category_num, false);
         cout<<"Out of bag validation confusion matrix: \n"<<oob_conf<<endl;
        
         if (valid_features.size() != 0) {
             vector<int> valid_predictions;
             for (int i = 0; i<valid_features.size(); i++) {
                 int pred = 0;
                 model.predict(valid_features[i], pred);
                 valid_predictions.push_back(pred);
             }
             assert(valid_predictions.size() == valid_labels.size());
             Eigen::MatrixXd valid_conf = DTUtil::confusionMatrix<int>(valid_predictions, valid_labels, category_num, false);
             Eigen::VectorXd accuracy = DTUtil::accuracyFromConfusionMatrix(valid_conf);
             cout<<"Validation confusion matrix: \n"<<valid_conf<<endl;
             cout<<"Validation accuracy: \n"<<accuracy.transpose()<<endl;
         }
    }
    return true;
}


bool DTClassifierBuilder::buildModel(DTClassifier & model,
                                     const vector< vector<VectorXf> > & features,
                                     const vector< vector<int> > & labels,
                                     const int max_num_frames,
                                     const char * model_file_name) const
{
    assert(features.size() == labels.size());
    
    model.tree_param_ = tree_param_;
    model.trees_.clear();   // @todo release memory
    
    
    const int tree_num = tree_param_.tree_num_;
    const int category_num = tree_param_.category_num_;
    for (int n = 0; n<tree_num; n++) {
        // randomly select frames
        vector<Eigen::VectorXf> train_features;
        vector<int> train_labels;
        for (int i = 0; i<max_num_frames; i++) {
            int rnd_idx = rand()%features.size();
            train_features.insert(train_features.end(), features[rnd_idx].begin(), features[rnd_idx].end());
            train_labels.insert(train_labels.end(), labels[rnd_idx].begin(), labels[rnd_idx].end());
        }
        
        vector<int> training_indices;
        for (int i = 0; i<train_features.size(); i++) {
            training_indices.push_back(i);
        }
        
        // training
        DTCTree * tree = new DTCTree();
        assert(tree);
        double tt = clock();
        tree->buildTree(train_features, train_labels, training_indices, tree_param_);
        model.trees_.push_back(tree);
        printf("build tree %d cost %lf minutes\n", n, (clock()- tt)/CLOCKS_PER_SEC/60.0);
        
        if (model_file_name != NULL) {
            model.save(model_file_name);
        }
        
        // single tree validataion error
        vector<Eigen::VectorXf> validation_features;
        vector<int> validation_labels;
        for (int i = 0; i<10; i++) {
            int rnd_idx = rand()%features.size();
            validation_features.insert(validation_features.end(), features[rnd_idx].begin(), features[rnd_idx].end());
            validation_labels.insert(validation_labels.end(), labels[rnd_idx].begin(), labels[rnd_idx].end());
        }
        
        vector<int> cv_predictions;
        for (int i = 0; i<validation_features.size(); i++) {
            int pred;
            bool is_pred = model.predict(validation_features[i], pred);
            assert(is_pred);
            if (is_pred) {
                cv_predictions.push_back(pred);
            }
        }
        
        Eigen::MatrixXd confusion = DTUtil::confusionMatrix<int>(cv_predictions, validation_labels, category_num, true);
        cout<<"cross validation confusion matrix: \n"<<confusion<<" from 10 images.\n";
    }
    
    printf("build model done %lu trees.\n", model.trees_.size());
    return true;
}









