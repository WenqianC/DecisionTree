//
//  BTDTRegressorBuilder.cpp
//  RGBD_RF
//
//  Created by jimmy on 2016-12-30.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#include "BTDTRegressorBuilder.h"
#include "DTRandom.h"
#include <iostream>

using std::cout;
using std::endl;


void BTDTRegressorBuilder::setTreeParameter(const BTDTRTreeParameter & param)
{
    tree_param_ = param;
}

bool BTDTRegressorBuilder::buildModel(BTDTRegressor & model,
                                      const vector<VectorXf> & features,
                                      const vector<VectorXf> & labels,
                                      const int maxCheck,
                                      const char * model_file_name) const
{
    model.reg_tree_param_ = tree_param_;
    model.trees_.clear();   // @todo release memory
    model.feature_dim_ = (int)features.front().size();
    model.label_dim_ = (int)labels.front().size();
    
    const int tree_num = tree_param_.tree_num_;
    for (int n = 0; n<tree_num; n++) {
        // bagging
        vector<unsigned int> training_indices;
        vector<unsigned int> validation_indices;
        DTRandom::outof_bag_sampling((unsigned int) features.size(), training_indices, validation_indices);
        
        TreeType * tree = new TreeType();
        assert(tree);
        double tt = clock();
        tree->buildTree(features, labels, training_indices, tree_param_);
        model.trees_.push_back(tree);
        
        printf("build tree %d cost %lf minutes\n", n, (clock()- tt)/CLOCKS_PER_SEC/60.0);
        
        // test on the validation data
        vector<Eigen::VectorXf> cv_errors;
        for (int i = 0; i<validation_indices.size(); i++) {
            const int index = validation_indices[i];
            Eigen::VectorXf pred;
            tree->predict(features[index], maxCheck, pred);
            cv_errors.push_back(pred - labels[index]);
        }
        
        Eigen::VectorXf cv_mean_error;
        Eigen::VectorXf cv_median_error;
        BTDTRUtil::mean_median_error<Eigen::VectorXf>(cv_errors, cv_mean_error, cv_median_error);
        cout<<"cross validation mean error: "<<cv_mean_error.transpose()<<"\n median error: "<<cv_median_error.transpose()<<endl;
        if (model_file_name != NULL) {
            model.save(model_file_name);
        }
    }
    printf("build model done %lu trees.\n", model.trees_.size());
    
    return true;
}

bool BTDTRegressorBuilder::buildModel(BTDTRegressor & model,
                                      const vector< vector<VectorXf> > & features,
                                      const vector< vector<VectorXf> > & labels,
                                      const int max_num_frames,
                                      const int maxCheck,
                                      const char * model_file_name) const
{
    assert(features.size() == labels.size());
    
    model.reg_tree_param_ = tree_param_;
    model.trees_.clear();   // @todo release memory
    model.feature_dim_ = (int)features[0].front().size();
    model.label_dim_   = (int)labels[0].front().size();
    
    const int tree_num = tree_param_.tree_num_;
    for (int n = 0; n<tree_num; n++) {
        // randomly select frames
        vector<Eigen::VectorXf> train_features;
        vector<Eigen::VectorXf> train_labels;
        for (int i = 0; i<max_num_frames; i++) {
            int rnd_idx = rand()%features.size();
            train_features.insert(train_features.end(), features[rnd_idx].begin(), features[rnd_idx].end());
            train_labels.insert(train_labels.end(), labels[rnd_idx].begin(), labels[rnd_idx].end());
        }
        
        vector<unsigned int> training_indices;
        for (int i = 0; i<train_features.size(); i++) {
            training_indices.push_back(i);
        }
        
        // training
        TreeType * tree = new TreeType();
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
        vector<Eigen::VectorXf> validation_labels;
        for (int i = 0; i<10; i++) {
            int rnd_idx = rand()%features.size();
            validation_features.insert(validation_features.end(), features[rnd_idx].begin(), features[rnd_idx].end());
            validation_labels.insert(validation_labels.end(), labels[rnd_idx].begin(), labels[rnd_idx].end());
        }
        
        vector<Eigen::VectorXf> cv_errors;
        for (int i = 0; i<validation_features.size(); i++) {
            Eigen::VectorXf pred;
            bool is_pred = tree->predict(validation_features[i], maxCheck, pred);
            if (is_pred) {
                cv_errors.push_back(pred - validation_labels[i]);
            }
        }
        
        Eigen::VectorXf cv_mean_error;
        Eigen::VectorXf cv_median_error;
        BTDTRUtil::mean_median_error<Eigen::VectorXf>(cv_errors, cv_mean_error, cv_median_error);
        cout<<"cross validation mean error: "<<cv_mean_error.transpose()<<"\n median error: "<<cv_median_error.transpose()<< "from 10 images.\n";
    }
    
    printf("build model done %lu trees.\n", model.trees_.size());
    return true;
}


static float minPredictionError(const Eigen::VectorXf& gd, const vector<Eigen::VectorXf>& predictions)
{
    assert(predictions.size() > 0);
    float minV = (gd - predictions[0]).norm();
    for (int i = 1; i<predictions.size(); i++) {
        float v = (gd - predictions[i]).norm();
        if (v < minV) {
            minV = v;
        }
    }
    return minV;
}

class FrameIndexError
{
public:
    unsigned int frame_idx_;
    float error_;
public:
    FrameIndexError(int index, float error)
    {
        frame_idx_ = index;
        error_ = error;
    }
    
    bool operator < (const FrameIndexError & other) const
    {
        return error_ < other.error_;
    }
};

bool BTDTRegressorBuilder::buildModel(BTDTRegressor & model,
                                      const vector< vector<VectorXf> > & features,
                                      const vector< vector<VectorXf> > & labels,
                                      const int sampleFrameNum,
                                      const int maxCheck,
                                      const float boostingRatio,
                                      const char * model_file_name) const
{
    assert(features.size() == labels.size());
    assert(boostingRatio >= 0 && boostingRatio <= 1.0);
    
    model.reg_tree_param_ = tree_param_;
    model.trees_.clear();   // @todo release memory
    model.feature_dim_ = (int)features[0].front().size();
    model.label_dim_   = (int)labels[0].front().size();
    
    const int tree_num = tree_param_.tree_num_;
    const int total_frame_num = (int)features.size();
    vector<unsigned int> learning_frames;      // learning frame index
    for (int i = 0; i<total_frame_num; i++) {
        learning_frames.push_back(i);
    }
    
    const int boost_frame_num  = sampleFrameNum * boostingRatio;
    const int random_frame_num = sampleFrameNum - boost_frame_num;
    
    vector<int> boost_frames;   // boost frame index, first boost frame index is randomly selected
    for (int i = 0; i<boost_frame_num; i++) {
        boost_frames.push_back(rand()%total_frame_num);
    }
    
    for (int n = 0; n<tree_num; n++) {
        std::random_shuffle(learning_frames.begin(), learning_frames.end());
        
        // randomly select frames
        vector<Eigen::VectorXf> train_features;
        vector<Eigen::VectorXf> train_labels;
        for (int i = 0; i<random_frame_num; i++) {
            int rnd_idx = learning_frames[i];
            train_features.insert(train_features.end(), features[rnd_idx].begin(), features[rnd_idx].end());
            train_labels.insert(train_labels.end(), labels[rnd_idx].begin(), labels[rnd_idx].end());
        }
        
        for (int i = 0; i<boost_frames.size(); i++) {
            int idx = boost_frames[i];
            train_features.insert(train_features.end(), features[idx].begin(), features[idx].end());
            train_labels.insert(train_labels.end(), labels[idx].begin(), labels[idx].end());
        }
        printf("train, random, boost frame number %ld %d %ld\n", random_frame_num + boost_frames.size(), random_frame_num, boost_frames.size());
        
        assert(train_features.size() == train_labels.size());
        
        vector<unsigned int> training_indices;
        for (int i = 0; i<train_features.size(); i++) {
            training_indices.push_back(i);
        }
        printf("training sample number %ld\n", training_indices.size());
        
        // training
        TreeType * tree = new TreeType();
        assert(tree);
        double tt = clock();
        tree->buildTree(train_features, train_labels, training_indices, tree_param_);
        model.trees_.push_back(tree);
        printf("build tree %d cost %lf minutes\n", n, (clock()- tt)/CLOCKS_PER_SEC/60.0);
        
        if (model_file_name != NULL) {
            model.save(model_file_name);
        }
        
        // test model on rest frames and measure frame level error
        //vector<float> frame_errors;
        vector<FrameIndexError> frameErrors;
        for (int i = 0; i<features.size(); i++) {
            int frame_index = i;
            vector<Eigen::VectorXf> cv_features = features[frame_index];
            vector<Eigen::VectorXf> cv_labels   = labels[frame_index];
            assert(cv_features.size() == cv_labels.size());
            
            vector<float> prediction_errors;
            for (int j = 0; j<cv_features.size(); j++) {
                vector<Eigen::VectorXf> predictions;
                model.predict(cv_features[j], maxCheck, predictions);
                float minError = minPredictionError(cv_labels[j], predictions);
                prediction_errors.push_back(minError);
            }
            std::sort(prediction_errors.begin(), prediction_errors.end());
           
            float median_error = prediction_errors[prediction_errors.size()/2];
            FrameIndexError fie(frame_index, median_error);
            frameErrors.push_back(fie);
        }
        
        // sort from large to small
        std::sort(frameErrors.begin(), frameErrors.end(),
                  [](const FrameIndexError& a, const FrameIndexError& b){return a.error_ > b.error_;});
        
        // print out first N/2 largest frame errors
        boost_frames.clear();
        printf("largest cross validation errors: \n");
        for (int i = 0; i<boost_frame_num; i++) {
            boost_frames.push_back(frameErrors[i].frame_idx_);
            printf("rank, error %d\t %f\n", i, frameErrors[i].error_);
        }
        // also add weights to these frames by duplicate frame index
        for (int i = 0; i<sampleFrameNum; i++) {
            learning_frames.push_back(frameErrors[i].frame_idx_);
        }
    }
    
    printf("build model done %lu trees.\n", model.trees_.size());
    return true;
}

