//
//  seq_dtr_util.cpp
//  Classifer_RF
//
//  Created by jimmy on 2017-02-25.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "seq_dtr_util.h"


void SeqDTRUtil::generateSequence(const vector<int> & fns,
                                  const vector<Eigen::VectorXf> & features,
                                  const vector<Eigen::VectorXf> & labels,
                                  vector<Eigen::MatrixXf>& time_seq_features,
                                  vector<Eigen::MatrixXf>& time_seq_labels,
                                  const int max_time_step,
                                  const int feature_number)
{
    assert(fns.size() == features.size());
    assert(fns.size() == labels.size());
    assert(max_time_step%2 == 1);
    assert(feature_number > 0);
    
    const int N = (int)fns.size();
    const int half_size = max_time_step/2;
    // randomly generate sequence
    const size_t feat_cols  = features[0].size();
    const size_t label_cols = labels[0].size();
    Eigen::MatrixXf long_feature = Eigen::MatrixXf::Zero(max_time_step, feat_cols);
    Eigen::MatrixXf long_label   = Eigen::MatrixXf::Zero(max_time_step, label_cols);
    for (int i = 0; i<feature_number; i++) {
        int rnd_index = (rand()%(N - 2 * max_time_step)) + max_time_step;
        assert(rnd_index >= max_time_step && rnd_index < N - max_time_step);
        if (fns[rnd_index - half_size] + max_time_step - 1 != fns[rnd_index + half_size]) {
            printf("filter %d %d, discontinuous sequence\n", fns[rnd_index - half_size], fns[rnd_index + half_size]);
            continue;
        }
        for (int j = - half_size; j <= half_size; j++) {
            int index = rnd_index + j;
            // feature
            long_feature.row(j + half_size) = features[index];
            long_label.row(j + half_size) = labels[index];
        }
        time_seq_features.push_back(long_feature);
        time_seq_labels.push_back(long_label);
    }
    assert(time_seq_features.size() == time_seq_labels.size());
    printf("generate %lu sequential samples, feature dimension %ld %ld\n", time_seq_features.size(), long_feature.rows(), long_feature.cols());
    printf("label dimension %ld %ld\n", long_label.rows(), long_label.cols());
}

void SeqDTRUtil::generateTestFeatures(const vector<int> & fns,
                                      const vector<Eigen::VectorXf> & features,
                                      vector<int>& time_seq_fns,          // output
                                      vector<Eigen::MatrixXf>& time_seq_features,   // output
                                      const int max_time_step,
                                      const int test_step)
{
    assert(fns.size() == features.size());
    assert(max_time_step%2 == 1);
    
    const int half_size = max_time_step/2;
    // sequentially generate sequence
    const size_t cols = features[0].size();
    Eigen::MatrixXf long_feature = Eigen::MatrixXf::Zero(max_time_step, cols);
    for (int i = half_size; i<fns.size() - half_size; i += test_step) {
        for (int j = - half_size; j <= half_size; j++) {
            int index = i + j;
            assert(index >= 0 && index < features.size());
            // feature
            long_feature.row(j + half_size) = features[index];
        }
        time_seq_fns.push_back(fns[i]);
        time_seq_features.push_back(long_feature);
    }
    assert(time_seq_fns.size() == time_seq_features.size());
    printf("generate %lu sequence from %lu features\n", time_seq_features.size(), features.size());
}
