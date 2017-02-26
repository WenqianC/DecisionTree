//
//  seq_dt_regressor.h
//  Classifer_RF
//
//  Created by jimmy on 2017-02-25.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __Classifer_RF__seq_dt_regressor__
#define __Classifer_RF__seq_dt_regressor__

#include <stdio.h>
#include <vector>
#include "seq_dtr_tree.h"

using std::vector;

class SeqDTRegressor
{
public:
    friend class SeqDTRegressorBuilder;
    
    typedef SeqDTRTreeNode Node;
    typedef SeqDTRTreeNode* NodePtr;
    typedef SeqDTRTree Tree;
    typedef SeqDTRTree* TreePtr;
    
    typedef SeqDTRTreeParameter TreeParameter;
    
    vector<TreePtr> trees_;
    //vector<double> weight_factor_;   // weight normalization factor to minimize bias
    TreeParameter reg_tree_param_;
    
    int feature_time_step_;   // max time step
    int feature_dim_;         // feature dimension
    int label_dim_;           // label dimension
    
public:
    SeqDTRegressor(){feature_time_step_ = 0, feature_dim_ = 0; label_dim_ = 0;}
    ~SeqDTRegressor(){}
    
    //bool computeWeightFactor();
    
    // assume the lenght of time steps is the same as the row of feature
    // raw prediction for one sequence, lower level time steps have large errors
    bool predict(const Eigen::MatrixXf & feature,
                 vector<Eigen::VectorXf> & predictions) const;
    
    // predictions: length is "feature_time_step"
    bool predict(const Eigen::MatrixXf & feature,
                 vector< vector<Eigen::VectorXf> > & predictions,
                 vector< vector<double> > & weights) const;
   
    
    /*
    // frame_numbers: continue with step 1
    // start_predict_frame_number_index: index of frame_numbers
    // max_time_step/2 frame are missed in the begining and ending of the sequence
    bool sequencePredict(const vector<int>& frame_numbers,
                         const vector<Eigen::VectorXf> & features,
                         vector<unsigned int> & predictions) const;
     */
    
    // frame_numbers: step 1, most of time (eg., 99% of time) continuous
    // for discontinuous frames, also report result
    bool multipleSequencePredict(const vector<int>& frame_numbers,
                                 const vector<Eigen::VectorXf> & features,
                                 vector<Eigen::VectorXf> & predictions) const;
     
   
    
    
    bool save(const char *fileName) const;
    bool load(const char *fileName);
     
    
};


#endif /* defined(__Classifer_RF__seq_dt_regressor__) */
