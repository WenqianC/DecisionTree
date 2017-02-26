//
//  seq_dt_regressor.cpp
//  Classifer_RF
//
//  Created by jimmy on 2017-02-25.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "seq_dt_regressor.h"
#include "seq_dtr_tree_node.h"

/*
bool SeqDTRegressor::computeWeightFactor()
{
    assert(trees_.size() > 0);
    weight_factor_.resize(feature_time_step_, 0.0);
    vector<int> count(feature_time_step_, 0);
    for (int i = 0; i<trees_.size(); i++) {
        TreePtr pTree = trees_[i];
        assert(pTree->time_steps_.size() == pTree->weights_.size());
        for (int j = 0; j<pTree->time_steps_.size(); j++) {
            int step = pTree->time_steps_[j];
            weight_factor_[step] += pTree->weights_[j];
            count[step]++;
        }
    }
    // average weight from multiple trees
    for (int i = 0; i<weight_factor_.size(); i++) {
        assert(count[i] > 0);
        weight_factor_[i] /= count[i];
    }
    for (int i = 0; i<weight_factor_.size(); i++) {
        printf("step %d, weight %lf\n", i, weight_factor_[i]);
    }
    return true;
}
 */
bool SeqDTRegressor::predict(const Eigen::MatrixXf & feature,
                             vector<Eigen::VectorXf> & output_predictions) const
{
    assert(trees_.size() > 0);
    const int max_time_step = (int)feature.rows();
    assert(feature_time_step_ == max_time_step);
    
    // every time step has multiple predictions (from multiple trees),
    vector< vector<Eigen::VectorXf> > all_preds(max_time_step);
    vector< vector<double> > all_weights(max_time_step);
    for (int n = 0; n < trees_.size(); n++) {
        vector<unsigned int> cur_time_steps;
        vector<double> cur_wts;
        vector<Eigen::VectorXf> cur_preds;
        
        bool is_pred = trees_[n]->predict(feature, cur_time_steps, cur_wts, cur_preds);
        if (is_pred) {
            for (int j = 0; j<cur_time_steps.size(); j++) {
                int step = cur_time_steps[j];
                assert(step >= 0 && step < all_preds.size());
                all_preds[step].push_back(cur_preds[j]);
                all_weights[step].push_back(cur_wts[j]);
            }
        }
    }
    
    const double epsilon = 0.00000001; // avoid divide zero
    // average all probability of each step
    for (int i = 0; i<all_preds.size(); i++) {
        if (all_preds[i].empty()) {
            printf("Warning: time step %d has no prediction\n", i);
            return false;
        }
        double wt_sum = epsilon;
        for (int j = 0; j<all_weights[i].size(); j++) {
            wt_sum += all_weights[i][j];
        }
        Eigen::VectorXf avg_pred = Eigen::VectorXf::Zero(label_dim_, 1);
        for (int j = 0; j<all_preds[i].size(); j++) {
            avg_pred += all_preds[i][j] * all_weights[i][j];
        }
        avg_pred /= wt_sum;
        output_predictions.push_back(avg_pred);
    }
    return true;
}

bool SeqDTRegressor::predict(const Eigen::MatrixXf & feature,
                             vector< vector<Eigen::VectorXf> > & predictions,
                             vector< vector<double> > & weights) const
{
    assert(trees_.size() > 0);
    const int max_time_step = (int)feature.rows();
    assert(feature_time_step_ == max_time_step);
    
    // every time step has multiple predictions (from multiple trees),
    predictions.resize(max_time_step);
    weights.resize(max_time_step);
    for (int n = 0; n < trees_.size(); n++) {
        vector<unsigned int> cur_time_steps;
        vector<double> cur_wts;
        vector<Eigen::VectorXf> cur_preds;
        
        bool is_pred = trees_[n]->predict(feature, cur_time_steps, cur_wts, cur_preds);
        if (is_pred) {
            for (int j = 0; j<cur_time_steps.size(); j++) {
                int step = cur_time_steps[j];
                assert(step >= 0 && step < predictions.size());
                predictions[step].push_back(cur_preds[j]);
                weights[step].push_back(cur_wts[j]);
            }
        }
    }
    
    for (int i = 0; i<predictions.size(); i++) {
        if (predictions[i].size() == 0) {
            printf("Warning: empty prediction at step %d\n", i);
            return false;
        }
    }
    return true;
}

bool SeqDTRegressor::multipleSequencePredict(const vector<int>& frame_numbers,
                                             const vector<Eigen::VectorXf> & features,
                                             vector<Eigen::VectorXf> & predictions) const
{
    assert(frame_numbers.size() == features.size());
    assert(trees_.size() > 0);
    assert(features.front().size() == feature_dim_);
    
    // check frame number
    for (int i = 0; i<frame_numbers.size() - 1; i++) {
        if(frame_numbers[i] + 1 != frame_numbers[i+1]) {
            printf("warning: discontinuous frame number %d %d, sub optimal for prediction.\n", frame_numbers[i], frame_numbers[i+1]);
        }
    }
    int max_time_step = feature_time_step_;
    assert(max_time_step%2 == 1);
    int half_size = max_time_step/2;
    
    // generate testing feature sequence
    vector<int> time_seq_fns;     // in the middle of time step
    vector<Eigen::MatrixXf> time_seq_features;
    SeqDTRUtil::generateTestFeatures(frame_numbers, features, time_seq_fns, time_seq_features, max_time_step, 1);
    assert(time_seq_fns.size() == time_seq_features.size());
    
    vector< vector <Eigen::VectorXf> > all_preds(time_seq_fns.size() + half_size * 2);
    vector< vector< double > >         all_weights(time_seq_fns.size() + half_size * 2);
    assert(all_preds.size() == frame_numbers.size());
    
    const int begining_index = half_size;
    for (int i = 0; i<time_seq_features.size(); i++) {
        vector< vector<Eigen::VectorXf> > cur_preds;
        vector< vector<double> > cur_weights;
        bool is_pred = this->predict(time_seq_features[i], cur_preds, cur_weights);
        assert(is_pred);
        for (int j = -half_size; j <= half_size; j++) {
            int index = i + j + begining_index;  // index of time sequence
            int index2 = j + half_size;          // index of current prediction
            all_preds[index].insert(all_preds[index].end(), cur_preds[index2].begin(), cur_preds[index2].end());
            all_weights[index].insert(all_weights[index].end(), cur_weights[index2].begin(), cur_weights[index2].end());
        }
    }
    
    // average prediction by weight
    const double epsilon = 0.00000001; // avoid divide zero
    for (int i = 0; i < all_weights.size(); i++) {
        if (all_weights[i].empty()) {
            printf("Warning: frame number %d has no prediction \n", frame_numbers[i]);
            return false;
        }
        assert(all_weights[i].size() > 0);
        double wt_sum = epsilon;
        for (int j = 0; j<all_weights[i].size(); j++) {
            wt_sum += all_weights[i][j];
        }
        Eigen::VectorXf avg_pred = Eigen::VectorXf::Zero(label_dim_, 1);
        for (int j = 0; j<all_preds[i].size(); j++) {
            avg_pred += all_preds[i][j] * all_weights[i][j];
        }
        avg_pred /= wt_sum;
        predictions.push_back(avg_pred);
    }
    assert(predictions.size() == frame_numbers.size());
    
    
    printf("predict %lu from %lu \n", predictions.size(), frame_numbers.size());
    return true;
}

bool SeqDTRegressor::save(const char *fileName) const
{
    assert(trees_.size() > 0);
    // write tree number and tree files to file Name
    FILE *pf = fopen(fileName, "w");
    if(!pf) {
        printf("Error: can not open file %s\n", fileName);
        return false;
    }
    fprintf(pf, "%d %d %d\n", feature_time_step_, feature_dim_, label_dim_);
    reg_tree_param_.writeToFile(pf);
    vector<string> tree_files;
    string baseName = string(fileName);
    baseName = baseName.substr(0, baseName.size()-4);
    for (int i = 0; i<trees_.size(); i++) {
        char buf[1024] = {NULL};
        sprintf(buf, "_%08d", i);
        string fileName = baseName + string(buf) + string(".txt");
        fprintf(pf, "%s\n", fileName.c_str());
        tree_files.push_back(fileName);
    }
    
    for (int i = 0; i<trees_.size(); i++) {
        if (trees_[i]) {
            SeqDTRTreeNode::writeTree(tree_files[i].c_str(), trees_[i]->time_steps_, trees_[i]->weights_, trees_[i]->root_);
        }
    }
    fclose(pf);
    printf("save to %s\n", fileName);
    return true;
}

bool SeqDTRegressor::load(const char *fileName)
{
    FILE *pf = fopen(fileName, "r");
    if (!pf) {
        printf("Error: can not open file %s\n", fileName);
        return false;
    }
    
    int ret_num = fscanf(pf, "%d %d %d", &feature_time_step_, &feature_dim_, &label_dim_);
    assert(ret_num == 3);
    
    bool is_read = reg_tree_param_.readFromFile(pf);
    assert(is_read);
    reg_tree_param_.printSelf();
    
    vector<string> treeFiles;
    for (int i = 0; i<reg_tree_param_.tree_num_; i++) {
        char buf[1024] = {NULL};
        fscanf(pf, "%s", buf);
        treeFiles.push_back(string(buf));
    }
    fclose(pf);
    
    for (int i = 0; i<trees_.size(); i++) {
        delete trees_[i];
        trees_[i] = 0;
    }
    trees_.clear();
    
    // read each tree
    for (int i = 0; i<treeFiles.size(); i++) {
        Node * root = NULL;
        bool isRead = false;
        vector<unsigned int> time_steps;
        vector<double> weights;
        isRead = Node::readTree(treeFiles[i].c_str(), time_steps, weights, root);
        assert(isRead);
        assert(root);
        
        TreePtr tree = new Tree();
        tree->root_ = root;
        tree->setTreeParameter(reg_tree_param_);
        tree->setTimesteps(time_steps);
        tree->setWeights(weights);
        trees_.push_back(tree);
    }
    printf("read from %s\n", fileName);
    
    return true;
}

