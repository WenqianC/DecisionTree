//
//  btdtr_ptz_builder.cpp
//  PTZBTRF
//
//  Created by jimmy on 2017-07-20.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "btdtr_ptz_builder.h"
#include "dt_util.hpp"
#include <iostream>

using std::cout;
using std::endl;


BTDTRPTZBuilder::BTDTRPTZBuilder()
{

}

BTDTRPTZBuilder::~BTDTRPTZBuilder()
{

}

void BTDTRPTZBuilder::setTreeParameter(const TreeParameter& param)
{
    tree_param_ = param;
}
   
bool BTDTRPTZBuilder::buildModel(BTDTRegressor& model,
                                 const vector<string> & feature_files,
                                 const vector<Eigen::Vector3f> & ptzs,
                                 const char *model_file_name) const
{
    assert(feature_files.size() == ptzs.size());
    
    model.trees_.clear();
    
    tree_param_.printSelf();
    
    const int frame_num = (int)feature_files.size();
    const int sampled_frame_num = std::min((int)feature_files.size(), tree_param_.sampled_frame_num_);
    const int tree_num = tree_param_.base_tree_param_.tree_num_;    
   
    const Eigen::Vector2f pp(tree_param_.pp_x_, tree_param_.pp_y_);
    for (int n = 0; n<tree_num; n++) {
        // randomly sample frames
        vector<string> sampled_files;
        vector<Eigen::VectorXf> sampled_ptzs;
        for (int j = 0; j<sampled_frame_num; j++) {
            int index = rand()%frame_num;
            sampled_files.push_back(feature_files[index]);
            sampled_ptzs.push_back(ptzs[index]);
        }
        
        printf("training from %lu frames\n", sampled_files.size());
        // sample from selected frames
        vector<VectorXf> features;
        vector<VectorXf> labels;
        for (int j = 0; j<sampled_files.size(); j++) {
            vector<btdtr_ptz_util::PTZSample> samples;
            btdtr_ptz_util::generatePTZSample(sampled_files[j].c_str(), pp, ptzs[j], samples);
            for (int k = 0; k< samples.size(); k++) {
                features.push_back(samples[k].descriptor_);
                labels.push_back(samples[k].pan_tilt_);
            }
        }
        assert(features.size() == labels.size());
        
        printf("training sample number is %lu\n", features.size());
        
        model.feature_dim_ = (int)features[0].size();
        model.label_dim_ = (int)labels[0].size();
        
        vector<unsigned int> indices = DTUtil::range<unsigned int>(0, (int)features.size(), 1);
        assert(indices.size() == features.size());
        
        TreePtr pTree = new TreeType();
        assert(pTree);
        double tt = clock();
        pTree->buildTree(features, labels, indices, tree_param_.base_tree_param_);
        printf("build a tree cost %lf seconds\n", (clock()-tt)/CLOCKS_PER_SEC );
        
        model.trees_.push_back(pTree);
        if (model_file_name != NULL) {            
            model.saveModel(model_file_name);
            printf("saved %s\n", model_file_name);
        }
        
        this->validationError(model, feature_files, ptzs, 2);
    }   
    return true;
}

bool BTDTRPTZBuilder::validationError(const BTDTRegressor & model,
                                      const vector<string> & feature_files,
                                      const vector<Eigen::Vector3f> & ptzs,
                                      const int sample_frame_num) const
{
    const Eigen::Vector2f pp(tree_param_.pp_x_, tree_param_.pp_y_);
   
    
    const int max_check = 8;
    // sample from selected frames
    for (int i = 0; i<sample_frame_num; i++) {
        int index = rand()%sample_frame_num;
        string feature_file_name = feature_files[index];
        Eigen::VectorXf ptz = ptzs[index];
        
        vector<btdtr_ptz_util::PTZSample> samples;
        btdtr_ptz_util::generatePTZSample(feature_file_name.c_str(), pp, ptz, samples);
        
        vector<Eigen::VectorXf> errors;
        for (int k = 0; k< samples.size(); k++) {
            Eigen::VectorXf feat = samples[k].descriptor_;
            Eigen::VectorXf label = samples[k].pan_tilt_;
            vector<Eigen::VectorXf> cur_predictions;
            vector<float> dist;
            model.predict(feat, max_check, cur_predictions, dist);
            long int min_v_index = std::distance(dist.begin(), std::min_element(dist.begin(), dist.end()));
            
            errors.push_back(cur_predictions[min_v_index] - label);
        }
        
        
        Eigen::VectorXf mean_error, median_error;
        DTUtil::meanMedianError(errors, mean_error, median_error);
        printf("tree number: %d, back tracking number %d\n", model.trees_.size(), max_check);
        cout<<"Validataion mean error: \n"<<mean_error.transpose()<<endl;
        cout<<"Validataion median error: \n"<<median_error.transpose()<<endl<<endl;
    }

    
    return true;
    
}
