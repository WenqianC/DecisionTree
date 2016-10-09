//
//  DTCUtil.h
//  Classifer_RF
//
//  Created by jimmy on 2016-07-26.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef __Classifer_RF__DTCUtil__
#define __Classifer_RF__DTCUtil__

#include <stdio.h>
#include <vector>
#include <Eigen/Dense>

using std::vector;
using Eigen::VectorXd;
using Eigen::MatrixXd;

class DTCTreeParameter
{
public:
    int tree_num_;               // number of trees
    int max_depth_;
    int min_leaf_node_num_;
    int min_split_num_;          // prevent too small node
    int split_candidate_num_;    // number of split in [v_min, v_max]
    bool verbose_;
    
    int feature_dimension_;          // feature dimension
    int category_num_;               // category number
    
    
    DTCTreeParameter()
    {
        tree_num_ = 5;
        max_depth_ = 10;
        min_leaf_node_num_ = 32;
        min_split_num_ = 16;
        split_candidate_num_ = 10;
        verbose_ = false;
        
        feature_dimension_ = 10;
        category_num_ = 2;
    }
    
    DTCTreeParameter(const int feat_dim, const int category_num)
    {
        tree_num_ = 5;
        max_depth_ = 10;
        min_leaf_node_num_ = 32;
        min_split_num_ = 16;
        split_candidate_num_ = 10;
        verbose_ = false;
        
        feature_dimension_ = feat_dim;
        category_num_ = category_num;
    }
};

class DTCSplitParameter
{
public:
    int split_dim_;
    double split_threshold_;
    double split_loss_;        // gini inpurity, cross entropy
    
    DTCSplitParameter()
    {
        split_dim_ = 0;
        split_threshold_ = 0.0;
        split_loss_ = 1.0;
    }
};

class DTCUtil
{
public:
    // generate N random value in [min_v, max_v]
    static vector<double>
    generateRandomNumber(const double min_v, const double max_v, int num);
    
    // -p log (p)
    static double
    crossEntropy(const VectorXd & prob);
    
    static bool
    isSameLabel(const vector<unsigned int> & labels, const vector<unsigned int> & indices);
    
    // confusion matrix of predictions
    static Eigen::MatrixXd
    confusionMatrix(const vector<Eigen::VectorXd> & probs, const vector<unsigned int> & labels);
};



  

#endif /* defined(__Classifer_RF__DTCUtil__) */
