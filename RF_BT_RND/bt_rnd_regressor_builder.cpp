//
//  bt_rnd_regressor_builder.cpp
//  RGBD_RF
//
//  Created by jimmy on 2017-01-21.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include <iostream>
#include "bt_rnd_regressor_builder.h"
#include "DTRandom.h"
#include "BTDTRUtil.h"
#include "cvx_io.hpp"

using std::cout;
using std::endl;

void BTRNDRegressorBuilder::setTreeParameter(const BTRNDTreeParameter & param)
{
    tree_param_ = param;
}

void BTRNDRegressorBuilder::setDatasetParameter(const DatasetParameter & param)
{
    dataset_param_ = param;
}

bool BTRNDRegressorBuilder::buildModel(Regressor & model,
                                       const vector<Feature> & features,
                                       const vector<VectorXf> & labels,
                                       const vector<cv::Mat> & rgb_images,
                                       const int max_check,
                                       const char * model_file_name) const
{
    model.reg_tree_param_ = tree_param_;
    model.trees_.clear();   // @todo release memory
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
        tree->buildTree(features, labels, training_indices, rgb_images, tree_param_);
        model.trees_.push_back(tree);
        
        printf("build tree %d cost %lf minutes\n", n, (clock()- tt)/CLOCKS_PER_SEC/60.0);
        
        // test on the validation data
        vector<float> cv_errors;
        for (int i = 0; i<validation_indices.size(); i++) {
            const int index = validation_indices[i];
            const int img_index = features[index].image_index_;
            Eigen::VectorXf pred;
            float dist = 0.0f;
            tree->predict(features[index], rgb_images[img_index], max_check, pred, dist);
            cv_errors.push_back((pred - labels[index]).norm());
        }        
        
        std::sort(cv_errors.begin(), cv_errors.end());
        cout<<"cross validation median error: "<<cv_errors[cv_errors.size()/2]<<endl;
        if (model_file_name != NULL) {
            model.save(model_file_name);
        }
    }
    printf("build model done %lu trees.\n", model.trees_.size());
    return true;
}

bool BTRNDRegressorBuilder::buildModel(Regressor& model,
                                       const vector<string> & rgb_img_files,
                                       const vector<string> & depth_img_files,
                                       const vector<string> & pose_files,
                                       const int max_check,
                                       const bool release_memory,
                                       const char *model_file_name) const
{
    assert(rgb_img_files.size() == depth_img_files.size());
    assert(rgb_img_files.size() == pose_files.size());
    
    dataset_param_.printSelf();
    tree_param_.printSelf();
    
    const int total_frame_num = (int)rgb_img_files.size();
    const int sampled_frame_num = std::min((int)rgb_img_files.size(), tree_param_.max_frame_num_);
    const int tree_num = tree_param_.tree_num_;
    const int sample_per_frame = tree_param_.sampler_num_per_frame_;
    const bool is_use_depth = tree_param_.is_use_depth_;
    const int wh_single_dim = tree_param_.wh_kernel_size_;
    
    model.trees_.resize(tree_num);
    for (int i = 0; i<model.trees_.size(); i++){
        model.trees_[i] = NULL;
    }
    
    model.reg_tree_param_    = tree_param_;
    model.dataset_param_ = dataset_param_;
    
    // default depth parameter of the dataset
    /*
    const int pyramid_level = 3;
    const double default_depth = 1.0;
    const double min_depth = 0.5;
    const double max_depth = 3.0;
     */
    
    using FeatureType = SCRFRandomFeature;
    using TreeType = BTRNDTree;
    // prepare data for training
    for (int n = 0; n<tree_num; n++){
        //randomly sample frames
        vector<string> sampled_rgb_files;
        vector<string> sampled_depth_files;
        vector<string> sampled_pose_files;
        for(int j = 0; j<sampled_frame_num; j++) {
            int index = rand()%total_frame_num;
            sampled_rgb_files.push_back(rgb_img_files[index]);
            sampled_depth_files.push_back(depth_img_files[index]);
            sampled_pose_files.push_back(pose_files[index]);
        }
        
        printf("training from %lu frames\n", sampled_rgb_files.size());
        
        // sample from selected frames
        vector<cv::Mat> rgb_images;
        vector<FeatureType> features;
        vector<Eigen::VectorXf> labels;
        for (int j = 0; j<sampled_rgb_files.size(); j++) {
            const char *rgb_img_file   = sampled_rgb_files[j].c_str();
            const char *depth_img_file = sampled_depth_files[j].c_str();
            const char *pose_file      = sampled_pose_files[j].c_str();
            
            cv::Mat rgb_img;
            CvxIO::imread_rgb_8u(rgb_img_file, rgb_img);
            vector<FeatureType> cur_features;
            vector<Eigen::VectorXf> cur_labels;            
            BTRNDUtil::randomSampleFromRgbdImages(rgb_img_file, depth_img_file, pose_file,
                                                  sample_per_frame, j, dataset_param_,
                                                  is_use_depth, false,
                                                  cur_features, cur_labels);
            
            BTRNDUtil::extractWHFeatureFromRgbImages(rgb_img_file, cur_features, wh_single_dim, false);
            
            features.insert(features.end(), cur_features.begin(), cur_features.end());
            labels.insert(labels.end(), cur_labels.begin(), cur_labels.end());
            rgb_images.push_back(rgb_img);
        }
        assert(features.size() == labels.size());
        
        printf("training sample number is %lu\n", features.size());
        
        vector<unsigned int> training_data_indices;
        for(int j = 0; j<features.size(); j++){
            training_data_indices.push_back(j);
        }
        
        assert(training_data_indices.size() == features.size());
        
        TreeType *tree = new TreeType();
        assert(tree);
        double tt = clock();
        tree->buildTree(features, labels, training_data_indices, rgb_images, tree_param_);
        printf("build a tree cost %lf minutes\n", (clock()- tt)/CLOCKS_PER_SEC/60.0);
        model.trees_[n] = tree;
        
        
        // single tree training error
        vector<float> training_errors;
        for(int j = 0; j<training_data_indices.size(); j += 10) {
            int index = training_data_indices[j];
            Eigen::VectorXf pred;
            float dist = 0.0f;
            bool isPredict = tree->predict(features[index], rgb_images[features[index].image_index_], max_check, pred, dist);
            if(isPredict) {
                training_errors.push_back( (pred - labels[index]).norm());
            }
        }
        
        // median train error
        std::sort(training_errors.begin(), training_errors.end());
        double median_train_error = training_errors[training_errors.size()/2];
        printf("tree %d\n. median training error is %lf meter\n", n, median_train_error);
        
        
        if(model_file_name != NULL) {
            model.save(model_file_name);
            printf("saved %s\n", model_file_name);
        }
        
        // validation error from a single tree
        this->testValidataionError(*tree, rgb_img_files, depth_img_files, pose_files, 10, max_check, 0.1);
        
        if(release_memory) {
            delete tree;
            tree = NULL;
            model.trees_[n] = NULL;
        }
    }
    
    return true;
}

bool BTRNDRegressorBuilder::testValidataionError(const BTRNDTree & tree,
                                                 const vector<string> & rgb_img_files,
                                                 const vector<string> & depth_img_files,
                                                 const vector<string> & pose_files,
                                                 const int sample_frame_num,
                                                 const int max_check,
                                                 const double error_threshold) const
{
    using FeatureType = SCRFRandomFeature;
    using TreeType = BTRNDTree;
    
    const int sample_per_frame = tree_param_.sampler_num_per_frame_;
    const bool is_use_depth = tree_param_.is_use_depth_;
    const int wh_single_dim = tree_param_.wh_kernel_size_;
    
    vector<string> sampled_rgb_files;
    vector<string> sampled_depth_files;
    vector<string> sampled_pose_files;
    for(int j = 0; j<sample_frame_num; j++) {
        int index = rand()%rgb_img_files.size();
        sampled_rgb_files.push_back(rgb_img_files[index]);
        sampled_depth_files.push_back(depth_img_files[index]);
        sampled_pose_files.push_back(pose_files[index]);
    }
    
    printf("cross validation from %lu frames\n", sampled_rgb_files.size());
    
    vector<int> backtracking;
    backtracking.push_back(1);
    backtracking.push_back(4);
    backtracking.push_back(max_check/2);
    backtracking.push_back(max_check);
    // sample from selected frames
    for (int i = 0; i<sampled_rgb_files.size(); i++) {
        const char *rgb_img_file   = sampled_rgb_files[i].c_str();
        const char *depth_img_file = sampled_depth_files[i].c_str();
        const char *pose_file      = sampled_pose_files[i].c_str();
        
        cv::Mat rgb_img;
        CvxIO::imread_rgb_8u(rgb_img_file, rgb_img);
        vector<FeatureType> cur_features;
        vector<Eigen::VectorXf> cur_labels;
        BTRNDUtil::randomSampleFromRgbdImages(rgb_img_file, depth_img_file, pose_file,
                                              sample_per_frame, i, dataset_param_,
                                              is_use_depth, false,
                                              cur_features, cur_labels);
        BTRNDUtil::extractWHFeatureFromRgbImages(rgb_img_file, cur_features, wh_single_dim, false);
        
        
        // test on each frame
        vector<float> median_errors;
        vector<float> percentages;
        for (int b = 0; b<backtracking.size(); b++) {
            vector<float> cv_errors;
            for(int j = 0; j<cur_features.size(); j++) {
                Eigen::VectorXf pred;
                float dist = 0.0f;
                bool isPredict = tree.predict(cur_features[j], rgb_img, backtracking[b], pred, dist);
                if(isPredict) {
                    cv_errors.push_back( (pred - cur_labels[j]).norm());
                }
            }
            
            // median cross validation error
            std::sort(cv_errors.begin(), cv_errors.end());
            double median_error = cv_errors[cv_errors.size()/2];
            median_errors.push_back(median_error);
            
            int inlier_num = 0;
            for (int j = 0; j < cv_errors.size(); j++) {
                if (cv_errors[j] < error_threshold) {
                    inlier_num++;
                }
            }
            percentages.push_back(1.0*inlier_num/cv_errors.size());       
        }
        
        assert(backtracking.size() == median_errors.size());
        assert(backtracking.size() == percentages.size());
        //output result
        for (int b = 0; b<backtracking.size(); b++) {
            printf("max check %d, median CV error: %lf meter\n", backtracking[b], median_errors[b]);
        }
        printf("\n");
        
        for (int b = 0; b<backtracking.size(); b++) {
            printf("max check %d, inlier percentage %lf within threshold %lf \n", backtracking[b], percentages[b], error_threshold);
        }
        printf("\n\n");
    }
    
    
    return true;
}


