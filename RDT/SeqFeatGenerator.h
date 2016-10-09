//
//  SeqFeatGenerator.h
//  Classifer_RF
//
//  Created by jimmy on 2016-10-07.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef __Classifer_RF__SeqFeatGenerator__
#define __Classifer_RF__SeqFeatGenerator__

// sequence feature generator

#include <stdio.h>
#include <vector>
#include <Eigen/Dense>

using std::vector;
using Eigen::VectorXd;

class SeqFeatGenerator
{
protected:
    vector<int> fns_;
    vector<Eigen::VectorXd > features_;
    vector<Eigen::VectorXd> labels_;
    
    vector<int> feature_sample_index_; // in increasing order, like 0, 10, 20 ...
    vector<int> label_sample_index_;   // in increasing roder, like 1, 2 ...
public:
    SeqFeatGenerator(){}
    ~SeqFeatGenerator(){}
    
    // set data, the frame numbers (fns) can be un continuous
    void setData(const vector<int> & fns,
                 const vector<Eigen::VectorXd > & features,
                 const vector<Eigen::VectorXd> & labels)
    {
        fns_      = fns;
        features_ = features;
        labels_   = labels;
    }
    
    void setLookbackIndices(const vector<int> & feature_index, const vector<int> & label_index);
    
    // alpha: weight of original label, 1.0 - alpha: predicted label
    bool generateFeatures(const vector<Eigen::VectorXd> & predicted_labels,
                          const double alpha,
                                   // output
                          vector<Eigen::VectorXd> & searn_features,
                          vector<Eigen::VectorXd> & searn_labels,
                          vector<int> & original_fn_index) const;
    
    bool generateLatestFeature(const vector<Eigen::VectorXd> & predicted_labels,
                               Eigen::VectorXd & feature) const;
    
    int maxLookback(void) const;
    
};

#endif /* defined(__Classifer_RF__SeqFeatGenerator__) */
