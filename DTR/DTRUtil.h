//
//  DTRUtil.h
//  Classifer_RF
//
//  Created by jimmy on 2016-10-03.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef __Classifer_RF__DTRUtil__
#define __Classifer_RF__DTRUtil__

#include <stdio.h>
#include <vector>
#include <Eigen/Dense>

using std::vector;
using Eigen::VectorXd;

class DTRTreeParameter
{
public:
    int tree_num_;               // number of trees
    int max_depth_;              // tree max depth
    int min_leaf_node_num_;      //
    int min_split_num_;          // prevent too small node
    int split_candidate_num_;    // number of split in [v_min, v_max]
    bool verbose_;
    
    int feature_dimension_;       // feature dimension
    int label_dim_;               // label dimension number
    
    
    DTRTreeParameter()
    {
        tree_num_ = 5;
        max_depth_ = 10;
        min_leaf_node_num_ = 32;
        min_split_num_ = 16;
        split_candidate_num_ = 10;
        verbose_ = false;
        
        feature_dimension_ = 10;
        label_dim_ = 1;
    }
    
    DTRTreeParameter(const int feat_dim, const int label_dim)
    {
        tree_num_ = 5;
        max_depth_ = 10;
        min_leaf_node_num_ = 32;
        min_split_num_ = 16;
        split_candidate_num_ = 10;
        verbose_ = false;
        
        feature_dimension_ = feat_dim;
        label_dim_ = label_dim;
    }
};

class DTRSplitParameter
{
public:
    int split_dim_;
    double split_threshold_;
    double split_loss_;        // spatial variance
    
    DTRSplitParameter()
    {
        split_dim_ = 0;
        split_threshold_ = 0.0;
        split_loss_ = 1.0;
    }
};

class DTRUtil
{
public:
    static void mean_stddev(const vector<VectorXd> & labels,
                            const vector<unsigned int> & indices,
                            VectorXd & mean, VectorXd & sigma);
    
    // spatial variance loss
    static double spatial_variance(const vector<VectorXd> & labels,
                                   const vector<unsigned int> & indices);
    
    // median: separate channel
    static void mean_median_error(const vector<VectorXd> & errors,
                                  Eigen::VectorXd & mean,
                                  Eigen::VectorXd & median);
    
    
    
    
};





#endif /* defined(__Classifer_RF__DTRUtil__) */
