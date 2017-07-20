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
#include <assert.h>
#include <unordered_map>
#include <string>

using std::unordered_map;
using std::string;

class OTFITreeParameter
{
public:
    int tree_num_;               // number of trees
    int max_tree_depth_;        
    int min_leaf_node_num_;    
    
    int candidate_dim_num_;          // dimension in one feature
    int candidate_threshold_num_;    // number of split in [v_min, v_max]    
    
    int category_num_;           // category number
    bool verbose_;               // output training process    
    bool verbose_leaf_;
    
    
    OTFITreeParameter()
    {
        tree_num_ = 5;
        max_tree_depth_ = 10;
        min_leaf_node_num_ = 32;        
        
        candidate_dim_num_ = 5;
        candidate_threshold_num_ = 10;
        
        category_num_ = 2;
        verbose_ = false;
        verbose_leaf_ = false;
    }

    bool readFromFile(FILE *pf)
    {
        assert(pf);
        
        const int param_num = 8;
        unordered_map<std::string, double> imap;
        for(int i = 0; i<param_num; i++)
        {
            char s[1024] = {NULL};
            double val = 0;
            int ret = fscanf(pf, "%s %lf", s, &val);
            if (ret != 2) {
                printf("read tree parameter Error: %s %f\n", s, val);
                assert(ret == 2);
            }
            imap[string(s)] = val;
        }
        assert(imap.size() == param_num);
        
        tree_num_ = (int)imap[string("tree_num")];
        max_tree_depth_ = (int)imap[string("max_tree_depth")];
        
        min_leaf_node_num_ = (int)imap[string("min_leaf_node_num")];       
        candidate_dim_num_ = (int)imap[string("candidate_dim_num")];
        candidate_threshold_num_ = (int)imap[string("candidate_threshold_num")];        
        
        category_num_ = (int)imap[string("category_num")]; 
        verbose_ = (bool)imap[string("verbose")];
        verbose_leaf_ = (bool)imap[string("verbose_leaf")];
        return true;
    }

    
    bool writeToFile(FILE *pf)const
    {
        assert(pf);
        fprintf(pf, "tree_num %d\n", tree_num_);
        fprintf(pf, "max_tree_depth %d\n", max_tree_depth_);        
        fprintf(pf, "min_leaf_node_num %d\n", min_leaf_node_num_);        
        
        fprintf(pf, "candidate_dim_num %d\n", candidate_dim_num_);
        fprintf(pf, "candidate_threshold_num %d\n", candidate_threshold_num_);
        
        fprintf(pf, "category_num %d\n", category_num_);      
        
        fprintf(pf, "verbose %d\n", (int)verbose_);
        fprintf(pf, "verbose_leaf %d\n\n", (int)verbose_leaf_);
        return true;
    }

    void printSelf() const
    {
        writeToFile(stdout);
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
