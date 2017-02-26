//
//  seq_dtr_util.h
//  Classifer_RF
//
//  Created by jimmy on 2017-02-25.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __Classifer_RF__seq_dtr_util__
#define __Classifer_RF__seq_dtr_util__

#include <stdio.h>
#include <vector>
#include <unordered_map>
#include <string>
#include <assert.h>
#include <Eigen/Dense>

using std::vector;
using std::unordered_map;
using std::string;

class SeqDTRTreeParameter
{
public:
    int tree_num_;               // number of trees
    int max_tree_depth_;         // tree max depth, [0, max_depth)
    
    int min_leaf_node_;          // leaf node  sample number
    int min_split_node_;         // internal node sample number
    
    int candidate_dim_num_;       // dimension in one feature
    int candidate_threshold_num_;    // number of split in [v_min, v_max]
    
    int max_time_step_;           // time window size in the sequence
    
    // random sample ratio
    double sample_ratio_;
    bool re_weight_;                // re weight using mean accuracy from cross validation
    
    bool verbose_;
    bool verbose_leaf_;
    
    
    SeqDTRTreeParameter()
    {
        tree_num_ = 5;
        max_tree_depth_ = 10;
        
        min_leaf_node_ = 32;
        min_split_node_ = 8;
        
        candidate_dim_num_ = 6;
        candidate_threshold_num_ = 10;
        
        max_time_step_ = 121;      
        
        sample_ratio_ = 0.1;
        re_weight_ = true;
        
        verbose_ = false;
        verbose_leaf_ = false;
    }
    
    bool readFromFile(const char *fileName)
    {
        FILE *pf = fopen(fileName, "r");
        if (!pf) {
            printf("can not read from %s \n", fileName);
            return false;
        }
        
        this->readFromFile(pf);
        fclose(pf);
        return true;
    }
    
    bool readFromFile(FILE *pf)
    {
        assert(pf);
        
        const int param_num = 11;
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
        
        min_leaf_node_ = (int)imap[string("min_leaf_node")];
        min_split_node_ = (int)imap[string("min_split_node")];
        candidate_dim_num_ = (int)imap[string("candidate_dim_num")];
        candidate_threshold_num_ = (int)imap[string("candidate_threshold_num")];
        
        max_time_step_ = (int)imap[string("max_time_step")];
        sample_ratio_ = imap[string("sample_ratio")];
        re_weight_ = (bool)imap[string("re_weight")];
        
        
        verbose_ = (bool)imap[string("verbose")];
        verbose_leaf_ = (bool)imap[string("verbose_leaf")];
        return true;
    }
    
    bool writeToFile(FILE *pf)const
    {
        assert(pf);
        fprintf(pf, "tree_num %d\n", tree_num_);
        fprintf(pf, "max_tree_depth %d\n", max_tree_depth_);
        
        fprintf(pf, "min_leaf_node %d\n", min_leaf_node_);
        fprintf(pf, "min_split_node %d\n", min_split_node_);
        
        fprintf(pf, "candidate_dim_num %d\n", candidate_dim_num_);
        fprintf(pf, "candidate_threshold_num %d\n", candidate_threshold_num_);
       
        fprintf(pf, "max_time_step %d\n", max_time_step_);
        fprintf(pf, "sample_ratio %f\n", sample_ratio_);
        fprintf(pf, "re_weight %d\n", re_weight_);
        
        fprintf(pf, "verbose %d\n", (int)verbose_);
        fprintf(pf, "verbose_leaf %d\n\n", (int)verbose_leaf_);
        return true;
    }
    
    void printSelf() const
    {
        writeToFile(stdout);
    }
};


class SeqDTRSplitParameter
{
public:
    int split_time_step_;        // time step
    int split_dim_;
    double split_threshold_;
    double split_loss_;         // mean error
    
    SeqDTRSplitParameter()
    {
        split_time_step_ = 0;
        split_dim_ = 0;
        split_threshold_ = 0.0;
        split_loss_ = 1.0;
    }
};

class SeqDTRUtil
{
public:
    // randomly generate training/testing examples
    // max_time_step: odd number
    static void generateSequence(const vector<int> & fns,
                                 const vector<Eigen::VectorXf> & features,
                                 const vector<Eigen::VectorXf> & labels,
                                 vector<Eigen::MatrixXf>& time_seq_features,
                                 vector<Eigen::MatrixXf>& time_seq_labels,
                                 const int max_time_step,
                                 const int feature_number);
    
    static void generateTestFeatures(const vector<int> & fns,
                              const vector<Eigen::VectorXf> & features,
                              vector<int>& time_seq_fns,          // output
                              vector<Eigen::MatrixXf>& time_seq_features,   // output
                              const int max_time_step,
                              const int test_step);
};













#endif /* defined(__Classifer_RF__seq_dtr_util__) */
