//
//  otfi_util.h
//  SequentialRandomForest
//
//  Created by jimmy on 2017-07-16.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __SequentialRandomForest__otfi_util__
#define __SequentialRandomForest__otfi_util__

#include <stdio.h>

class OTFITreeParameter
{
public:
    int tree_num_;               // number of trees
    int max_tree_depth_;
    int min_leaf_node_num_;
    int min_split_num_;          // prevent too small node
    
    int candidate_dim_num_;          // dimension in one feature
    int candidate_threshold_num_;    // number of split in [v_min, v_max]
    
    
    int category_num_;           // category number
        
    bool verbose_leaf_;
    bool verbose_;               // output training process
    
    OTFITreeParameter()
    {
        tree_num_ = 5;
        max_tree_depth_ = 10;
        min_leaf_node_num_ = 32;
        min_split_num_ = 16;
        
        candidate_dim_num_ = 5;
        candidate_threshold_num_ = 10;
        
        category_num_ = 2;
        
        
        verbose_ = false;
        verbose_leaf_ = false;
    }
};

class OTFISplitParameter
{
public:
    int dim_;             //
    double threshold_;
    double lower_bound_;  // of training examples
    double upper_bound_;
    double loss_;        // gini inpurity, or cross entropy
    
    OTFISplitParameter()
    {
        dim_ = 0;
        threshold_ = 0.0;
        lower_bound_ = 0.0;
        upper_bound_ = 0.0;
        loss_ = 1.0;
    }
};



#endif /* defined(__SequentialRandomForest__otfi_util__) */
