//
//  SeqFeatGenerator.cpp
//  Classifer_RF
//
//  Created by jimmy on 2016-10-07.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#include "SeqFeatGenerator.h"



void SeqFeatGenerator::setLookbackIndices(const vector<int> & feature_index, const vector<int> & label_index)
{
    feature_sample_index_ = feature_index;
    label_sample_index_   = label_index;
    
    std::sort(feature_sample_index_.begin(), feature_sample_index_.end());
    std::sort(label_sample_index_.begin(), label_sample_index_.end());
    if (!label_sample_index_.empty() && label_sample_index_[0] < 0) {
        printf("label index must be positive, can not look forward.");
        assert(label_sample_index_[0] > 0); // not know the current label
    }
}

bool SeqFeatGenerator::generateFeatures(const vector<Eigen::VectorXd> & predicted_labels,
                                        const double alpha,
                                        // output
                                        vector<Eigen::VectorXd> & searn_features,
                                        vector<Eigen::VectorXd> & searn_labels,
                                        vector<int> & original_fn_index) const
{
    assert(fns_.size() == features_.size());
    assert(fns_.size() == labels_.size());
    assert(fns_.size() ==  predicted_labels.size());
    assert(fns_.size() > 0);
    assert(feature_sample_index_.size() > 0 || label_sample_index_.size() > 0); // at least look back in feature or in label
    assert(alpha >= 0 && alpha <= 1.0);
    
    searn_features.clear();
    searn_labels.clear();
    original_fn_index.clear();
    
    // collect all sample index (i.e frame number)
    vector<int> sample_index;
    sample_index.insert(sample_index.end(), feature_sample_index_.begin(), feature_sample_index_.end());
    sample_index.insert(sample_index.end(), label_sample_index_.begin(), label_sample_index_.end());
    assert(sample_index.size() > 0);
    
    int max_look_back = *std::max_element(sample_index.begin(), sample_index.end());
    assert(max_look_back > 0);
    
    const int org_feature_dim = (int)features_.front().size();
    const int org_label_dim = (int)labels_.front().size();
    const int feature_length  = org_feature_dim * (int)feature_sample_index_.size() + org_label_dim * (int)label_sample_index_.size();
        
    Eigen::VectorXd concat_feature(feature_length);
    for (int i = max_look_back; i <fns_.size(); i++) {
        // check continuous
        bool isContinuous = true;
        for (int j = i-max_look_back; j < i; j++) {
            if (fns_[j] + 1 != fns_[j+1]) {
                isContinuous = false;
                break;
            }
        }
        if (!isContinuous) {
            // printf("feature frame not continuous\n");
            continue;
        }
        
        vector<Eigen::VectorXd> cur_feat_labels;
        // only generate continuous features
        for (int j = 0; j<feature_sample_index_.size(); j++) {
            int idx = i - feature_sample_index_[j];
            assert(idx >= 0 && idx < features_.size());
            cur_feat_labels.push_back(features_[idx]);
        }
        for (int j = 0; j<label_sample_index_.size(); j++) {
            int idx = i - label_sample_index_[j];
            assert(idx >= 0 && idx < labels_.size());
            cur_feat_labels.push_back(labels_[idx]);
        }
        int concat_idx = 0;
        for (int j = 0; j<cur_feat_labels.size(); j++) {
            for (int k = 0; k<cur_feat_labels[j].size(); k++) {
                concat_feature[concat_idx] = cur_feat_labels[j][k];
                concat_idx++;
            }
        }
        assert(concat_idx == feature_length);
        
        // generate labels
        // labels only used in mixture of label and prediction
        Eigen::VectorXd combined_label = alpha * labels_[i] + (1.0 - alpha) * predicted_labels[i];
        searn_features.push_back(concat_feature);
        searn_labels.push_back(combined_label);
        original_fn_index.push_back(i);
    }
    
    assert(searn_features.size() == searn_labels.size());
    assert(original_fn_index.size() <= fns_.size());
    printf("original, new feature number is (%lu, %lu)\n", features_.size(), searn_features.size());
    
    return true;
}

bool SeqFeatGenerator::generateLatestFeature(const vector<Eigen::VectorXd> & predicted_labels,
                                             Eigen::VectorXd & feature) const
{
    assert(fns_.size() == features_.size());
    assert(fns_.size() == labels_.size());
    
    assert(fns_.size() > 0);
    assert(feature_sample_index_.size() > 0 || label_sample_index_.size() > 0); // at least look back in feature or in label
    
    // collect all sample index (i.e frame number)    
    int max_look_back = this->maxLookback();
    assert(max_look_back > 0);
    
    const int org_feature_dim = (int)features_.front().size();
    const int org_label_dim = (int)labels_.front().size();
    const int feature_length  = org_feature_dim * (int)feature_sample_index_.size() + org_label_dim * (int)label_sample_index_.size();
    
    Eigen::VectorXd concat_feature(feature_length);
    // look at last feature
    for (int i = (int)predicted_labels.size() - 1; i < predicted_labels.size(); i++) {
        // check continuous
        bool isContinuous = true;
        for (int j = i-max_look_back; j < i; j++) {
            if (fns_[j] + 1 != fns_[j+1]) {
                isContinuous = false;
                break;
            }
        }
        if (!isContinuous) {
            return false;
        }        
        vector<Eigen::VectorXd> cur_feat_labels;
        // only generate continuous features
        for (int j = 0; j<feature_sample_index_.size(); j++) {
            int idx = i - feature_sample_index_[j];
            assert(idx >= 0 && idx < features_.size());
            cur_feat_labels.push_back(features_[idx]);
        }
        for (int j = 0; j<label_sample_index_.size(); j++) {
            int idx = i - label_sample_index_[j];
            assert(idx >= 0 && idx < predicted_labels.size());
            cur_feat_labels.push_back(predicted_labels[idx]);
        }
        int concat_idx = 0;
        for (int j = 0; j<cur_feat_labels.size(); j++) {
            for (int k = 0; k<cur_feat_labels[j].size(); k++) {
                concat_feature[concat_idx] = cur_feat_labels[j][k];
                concat_idx++;
            }
        }
        assert(concat_idx == feature_length);
        feature = concat_feature;
    }
    return true;
}

int SeqFeatGenerator::maxLookback(void) const
{
    vector<int> sample_index;
    sample_index.insert(sample_index.end(), feature_sample_index_.begin(), feature_sample_index_.end());
    sample_index.insert(sample_index.end(), label_sample_index_.begin(), label_sample_index_.end());
    assert(sample_index.size() > 0);
    
    int max_look_back = *std::max_element(sample_index.begin(), sample_index.end());
    assert(max_look_back > 0);
    return max_look_back;
}

