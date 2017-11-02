//
//  dtr_param.h
//  SequentialRandomForest
//
//  Created by jimmy on 2017-11-01.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __SequentialRandomForest__dtr_param__
#define __SequentialRandomForest__dtr_param__

#include <stdio.h>
#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include "dt_param_parser.h"

using std::vector;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::cout;
using std::endl;

class DTRTreeParameter
{
public:
    int tree_num_;               // number of trees
    int max_depth_;
    
    int min_leaf_node_num_;
    int min_split_num_;          // prevent too small node
    int split_candidate_num_;    // number of split in [v_min, v_max]
    
    int feature_dimension_;      // feature dimension
    
    bool verbose_leaf_;
    bool verbose_;               // output training process
    
    DTRTreeParameter()
    {
        tree_num_ = 5;
        max_depth_ = 10;
        min_leaf_node_num_ = 32;
        min_split_num_ = 16;
        split_candidate_num_ = 10;
        
        feature_dimension_ = 10;
        
        verbose_ = false;
        verbose_leaf_ = false;
    }
    
    
    
    bool readFromFile(const char *fileName)
    {
        dt::ParameterParser parser;
        bool is_read = parser.loadParameter(fileName);
        assert(is_read);
        
        parser.getIntValue("tree_num", tree_num_);
        parser.getIntValue("max_depth", max_depth_);
        parser.getIntValue("min_leaf_node_num", min_leaf_node_num_);
        parser.getIntValue("min_split_num", min_split_num_);
        parser.getIntValue("split_candidate_num", split_candidate_num_);
        parser.getIntValue("feature_dimension", feature_dimension_);
        
        parser.getBoolValue("verbose", verbose_);
        parser.getBoolValue("verbose_leaf", verbose_leaf_);
        return true;
    }
    
    bool readFromFile(FILE *pf)
    {
        dt::ParameterParser parser;
        bool is_read = parser.readFromFile(pf);
        assert(is_read);
        
        parser.getIntValue("tree_num", tree_num_);
        parser.getIntValue("max_depth", max_depth_);
        parser.getIntValue("min_leaf_node_num", min_leaf_node_num_);
        parser.getIntValue("min_split_num", min_split_num_);
        parser.getIntValue("split_candidate_num", split_candidate_num_);
        parser.getIntValue("feature_dimension", feature_dimension_);
        parser.getBoolValue("verbose", verbose_);
        parser.getBoolValue("verbose_leaf", verbose_leaf_);
        return true;
    }
    
    bool writeToFile(FILE *pf)const
    {
        dt::ParameterParser parser;
        parser.setIntValue("tree_num", tree_num_);
        parser.setIntValue("max_depth", max_depth_);
        parser.setIntValue("min_leaf_node_num", min_leaf_node_num_);
        parser.setIntValue("min_split_num", min_split_num_);
        
        parser.setIntValue("split_candidate_num", split_candidate_num_);
        parser.setIntValue("feature_dimension", feature_dimension_);
        
        
        parser.setBoolValue("verbose", verbose_);
        parser.setBoolValue("verbose_leaf", verbose_leaf_);
        
        parser.writeToFile(pf);
        return true;
    }
    
    friend std::ostream& operator <<(std::ostream& os, const DTRTreeParameter & p)
    {
        os<<"tree number: "<<p.tree_num_<<endl;
        os<<"max_depth  : "<<p.max_depth_<<endl;
        os<<"min_leaf_node: "<<p.min_leaf_node_num_<<endl;
        os<<"min_split_num: "<<p.min_split_num_<<endl;
        os<<"split_candidate_num: "<<p.split_candidate_num_<<endl;
        os<<"feature_dimension: "<<p.feature_dimension_<<endl;
        return os;
    }
};

class DTRSplitParameter
{
public:
    int dim_;
    double threshold_;
    double loss_;        // gini inpurity, cross entropy
    
    DTRSplitParameter()
    {
        dim_ = 0;
        threshold_ = 0.0;
        loss_ = 1.0;
    }
};



#endif /* defined(__SequentialRandomForest__dtr_param__) */
