//
//  bt_dtc_param.h
//  PTZBTRF
//
//  Created by jimmy on 2017-10-13.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __PTZBTRF__bt_dtc_param__
#define __PTZBTRF__bt_dtc_param__

#include <stdio.h>
#include <vector>
#include <Eigen/Dense>
#include <unordered_map>
#include <string>
#include "dt_param_parser.h"

using std::vector;
using std::string;
using Eigen::VectorXf;
using Eigen::VectorXd;

class BTDTCTreeParameter
{
public:
    int tree_num_;               // number of trees
    int max_tree_depth_;         // tree max depth
    int max_balanced_depth_;     // 0 - max_balanced_tree_depth, encourage balanced tree instead of smaller entropy
    
    int min_leaf_size_;          // leaf node  sample number
    int min_split_size_;         // internal node sample number
    
    int candidate_dim_num_;
    int candidate_threshold_num_;    // number of split in [v_min, v_max]
    int category_num_;
    
    bool verbose_;
    bool verbose_leaf_;
    
    BTDTCTreeParameter()
    {
        tree_num_ = 5;
        max_tree_depth_ = 10;
        max_balanced_depth_ = -1;  // default is no balance level
        
        min_leaf_size_ = 32;
        min_split_size_ = 8;
        
        candidate_dim_num_ = 6;
        candidate_threshold_num_ = 10;
        category_num_ = 2;
        
        verbose_ = false;
        verbose_leaf_ = false;
    }
    
    void setParser(dt::ParameterParser & parser) const
    {
        parser.setIntValue("tree_num", tree_num_);
        parser.setIntValue("max_tree_depth", max_tree_depth_);
        parser.setIntValue("max_balanced_depth", max_balanced_depth_);
        parser.setIntValue("min_leaf_size", min_leaf_size_);
        parser.setIntValue("min_split_size", min_split_size_);
        
        parser.setIntValue("candidate_dim_num", candidate_dim_num_);
        parser.setIntValue("candidate_threshold_num", candidate_threshold_num_);
        parser.setIntValue("category_num", category_num_);
        
        parser.setBoolValue("verbose", verbose_);
        parser.setBoolValue("verbose_leaf", verbose_leaf_);
    }
    
    void getParameterFromParser(const dt::ParameterParser & parser)
    {
        parser.getIntValue("tree_num", tree_num_);
        parser.getIntValue("max_tree_depth", max_tree_depth_);
        parser.getIntValue("max_balanced_depth", max_balanced_depth_);
        parser.getIntValue("min_leaf_size", min_leaf_size_);
        parser.getIntValue("min_split_size", min_split_size_);
        
        parser.getIntValue("candidate_dim_num", candidate_dim_num_);
        parser.getIntValue("candidate_threshold_num", candidate_threshold_num_);
        parser.getIntValue("category_num", category_num_);
        
        parser.getBoolValue("verbose", verbose_);
        parser.getBoolValue("verbose_leaf", verbose_leaf_);
    }
    
    bool readFromFile(const char *fileName)
    {
        dt::ParameterParser parser;
        bool is_read = parser.loadParameter(fileName);
        assert(is_read);
        getParameterFromParser(parser);
        
        return true;
    }
    
    bool readFromFile(FILE *pf)
    {
        assert(pf);
        
        dt::ParameterParser parser;
        bool is_read = parser.readFromFile(pf);
        assert(is_read);
        getParameterFromParser(parser);
        
        return true;
    }
    
    bool writeToFile(FILE *pf)const
    {
        dt::ParameterParser parser;
        setParser(parser);
        
        parser.writeToFile(pf);
        return true;
    }
    
    void printSelf() const
    {
        dt::ParameterParser parser;
        setParser(parser);
        parser.printSelf();
    }
};

struct BTDTCSplitParameter
{
public:
    int split_dim_;         // dimension in the feature
    double split_threshold_; // a threshold in feature space
    
    // auxilary data
    double split_loss_;    // loss of split, e.g., spatial variance
    BTDTCSplitParameter()
    {
        split_dim_ = 0;
        split_threshold_ = 0;
        split_loss_ = INT_MAX;
    }
};

#endif /* defined(__PTZBTRF__bt_dtc_param__) */
