//
//  BTDTRNode.h
//  RGBD_RF
//
//  Created by jimmy on 2016-12-29.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef __RGBD_RF__BTDTRNode__
#define __RGBD_RF__BTDTRNode__

// back tracking decision tree Node for regression
// idea: during the testing, back tracking trees once the testing example reaches the leaf node. It "should" increase performance
// disadvantage: increasing storage of the model, decrease speed in testing

#include <stdio.h>
#include <vector>
#include <Eigen/Dense>
#include "BTDTRUtil.h"

using std::vector;
using Eigen::VectorXf;

class BTDTRNode
{
public:
    BTDTRNode *left_child_;
    BTDTRNode *right_child_;
    int depth_;
    bool is_leaf_;
    
    // split parameter
    BTDTRSplitParameter split_param_;
    int sample_num_;
    double sample_percentage_;  // sample percentage of parent node
    
    // leaf node property
    VectorXf label_mean_;      // label
    VectorXf label_stddev_;
    VectorXf feat_mean_;
    int index_;               // @to leaf node index, in-order traversal
    
    
    typedef BTDTRNode* NodePtr;
    typedef BTDTRSplitParameter SplitParameter;
    
public:
    BTDTRNode(int depth)
    {
        left_child_ = NULL;
        right_child_ = NULL;
        depth_   = depth;
        is_leaf_ = false;
        sample_num_ = 0;
        sample_percentage_ = 0.0;
        index_ = -1;
    }
    
    static bool writeTree(const char *fileName, const NodePtr root, const int leafNodeNum);
    static bool readTree(const char *fileName, NodePtr & root, int &leafNodeNum);
    
private:
    static void writeNode(FILE *pf, const NodePtr node);
    static void readNode(FILE *pf, NodePtr & node);
    
};

#endif /* defined(__RGBD_RF__BTDTRNode__) */
