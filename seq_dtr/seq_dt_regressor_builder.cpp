//
//  seq_dt_regressor_builder.cpp
//  Classifer_RF
//
//  Created by jimmy on 2017-02-25.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "seq_dt_regressor_builder.h"
#include "seq_dtr_tree.h"
#include "dt_util.h"
#include <iostream>

using Eigen::MatrixXf;

using std::cout;
using std::endl;

void SeqDTRegressorBuilder::setTreeParameter(const SeqDTRTreeParameter & param)
{
    tree_param_ = param;
}

bool SeqDTRegressorBuilder::buildModel(SeqDTRegressor & model,
                                        const vector<int>& frame_numbers,
                                        const vector<Eigen::VectorXf> & features,
                                        const vector<Eigen::VectorXf> & labels,                                        
                                        const char * model_file_name) const
{
    assert(frame_numbers.size() == features.size());
    assert(frame_numbers.size() == labels.size());
    
    const int max_time_step = tree_param_.max_time_step_;
    assert(max_time_step%2 == 1);
    
    model.reg_tree_param_ = tree_param_;
    model.trees_.clear();   // @todo release memory
    model.feature_time_step_ = max_time_step;
    model.feature_dim_ = (int)features.front().size();
    model.label_dim_ = (int)labels[0].size();
    
    const double sample_ratio = tree_param_.sample_ratio_;
    const int sampled_num = features.size() * sample_ratio;
    const int tree_num = tree_param_.tree_num_;
    const int tree_depth = tree_param_.max_tree_depth_;
    const bool re_weight = tree_param_.re_weight_;
    
    
    // randomly sample time steps
    vector<unsigned int> time_step_candidates;
    while (time_step_candidates.size() < tree_depth * tree_num) {
        vector<unsigned int> possible_time_steps = DTUtil::range<unsigned int>(0, max_time_step, 1);
        std::random_shuffle(possible_time_steps.begin(), possible_time_steps.end());
        time_step_candidates.insert(time_step_candidates.end(), possible_time_steps.begin(), possible_time_steps.end());
    }
    
    
    for (int n = 0; n<tree_num; n++) {
        // sampling
        vector<Eigen::MatrixXf> sampled_features;
        vector<Eigen::MatrixXf> sampled_labels;
        SeqDTRUtil::generateSequence(frame_numbers, features, labels, sampled_features, sampled_labels, max_time_step, sampled_num);
        
        printf("sample number: %lu, ratio: %f\n", sampled_features.size(), 1.0 * sampled_features.size()/features.size());
        assert(sampled_features.size() == sampled_labels.size());
        
        vector<unsigned int> training_indices;
        vector<unsigned int> validation_indices;
        DTRandom::outof_bag_sampling((unsigned int) sampled_features.size(), training_indices, validation_indices);
        
        vector<unsigned int> cur_time_steps(time_step_candidates.begin() + n * tree_depth, time_step_candidates.begin() + (n + 1)*tree_depth);
        std::sort(cur_time_steps.begin(), cur_time_steps.end());
        
        // build tree
        SeqDTRTree * tree = new SeqDTRTree();
        assert(tree);
        double tt = clock();
        tree->buildTree(sampled_features, sampled_labels, training_indices, tree_param_, cur_time_steps);
        
        
        // single tree cross validation
        vector< vector<Eigen::VectorXf> > validation_error_seqs(cur_time_steps.size());
        for (int i = 0; i<validation_indices.size(); i++) { // sample index
            const int index = validation_indices[i];
            vector<unsigned int> time_steps;
            vector<Eigen::VectorXf> predictions;       // a sequence of predictions
            bool is_pred = tree->rawPredict(sampled_features[index], time_steps, predictions);
            assert(is_pred);
            for (int j = 0; j<predictions.size(); j++) {
                Eigen::VectorXf dif = sampled_labels[index].row(time_steps[j]) - predictions[j];
                validation_error_seqs[j].push_back(dif.array().abs());
            }
        }
        
        printf("single tree out of bag validation begin:\n");
        assert(labels[0].size() == 1);
        vector<double> weights;
        for (int i = 0; i<validation_error_seqs.size(); i++) {
            Eigen::VectorXf mean_error;
            Eigen::VectorXf median_error;
            DTUtil::meanMedianError(validation_error_seqs[i], mean_error, median_error);
            double wt = exp(-mean_error[0]);
            cout<<"tree depth "<<i<<" mean error: "<<median_error.transpose()<<" weight: "<<wt<<endl;
            weights.push_back(wt); // e(-x)
        }
        printf("single tree out of bag validation end.\n");
        tree->setWeights(weights);
        
        model.trees_.push_back(tree);
        printf("build tree %d cost %lf minutes\n", n, (clock()- tt)/CLOCKS_PER_SEC/60.0);        
        
        if (model_file_name != NULL) {
            model.save(model_file_name);
        }
    }
    
    return true;
}