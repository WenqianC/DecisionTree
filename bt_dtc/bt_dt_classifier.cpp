//
//  bt_dt_classifier.cpp
//  PTZBTRF
//
//  Created by jimmy on 2017-10-13.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "bt_dt_classifier.h"

BTDTClassifier::BTDTClassifier()
{
    feature_dim_ = 0;
    category_num_ = 0;
}
BTDTClassifier::~BTDTClassifier()
{
    //@todo, release memory in trees_
    
}
bool BTDTClassifier::predict(const Eigen::VectorXf & feature,
                             const int max_check,
                             vector<int> & predictions,
                             vector<float> & dists) const
             
{
    assert(trees_.size() > 0);
    assert(feature_dim_ == feature.size());
    assert(predictions.size() == 0);
    assert(dists.size() == 0);
    
    // predict from each tree
    for (int i = 0; i<trees_.size(); i++) {
        int cur_pred = 0;
        float dist = 0;
        bool is_pred = trees_[i]->predict(feature, max_check, cur_pred, dist);
        if (is_pred) {
            predictions.push_back(cur_pred);
            dists.push_back(dist);
        }
    }
    assert(predictions.size() == dists.size());
    return predictions.size() == trees_.size();
}

bool BTDTClassifier::save(const char *file_name) const
{
    return true;
}

bool BTDTClassifier::load(const char *file_name)
{
    return true;
}
