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
#include <unordered_map>
#include <string>

using std::vector;
using std::unordered_map;
using std::string;
using Eigen::VectorXd;

class DTRTreeParameter
{
public:
    int tree_num_;               // number of trees
    int max_tree_depth_;         // tree max depth
    
    int min_leaf_node_;          // leaf node  sample number
    int min_split_node_;         // internal node sample number
    
    int candidate_dim_num_;
    int candidate_threshold_num_;    // number of split in [v_min, v_max]
    
    bool verbose_;
    bool verbose_leaf_;
    
    
    DTRTreeParameter()
    {
        tree_num_ = 5;
        max_tree_depth_ = 10;
        
        min_leaf_node_ = 32;
        min_split_node_ = 8;
        
        candidate_dim_num_ = 6;
        candidate_threshold_num_ = 10;
        
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
        assert(imap.size() == 8);
        
        tree_num_ = (int)imap[string("tree_num")];
        max_tree_depth_ = (int)imap[string("max_tree_depth")];
        min_leaf_node_ = (int)imap[string("min_leaf_node")];
        min_split_node_ = (int)imap[string("min_split_node")];
        candidate_dim_num_ = (int)imap[string("candidate_dim_num")];
        candidate_threshold_num_ = (int)imap[string("candidate_threshold_num")];
        
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
        
        fprintf(pf, "verbose %d\n", (int)verbose_);
        fprintf(pf, "verbose_leaf %d\n\n", (int)verbose_leaf_);
        return true;
    }
    
    void printSelf() const
    {
        printf("RGB tree parameters:\n");
        printf("tree_num %d\n", tree_num_);
        printf("max_tree_depth %d\n", max_tree_depth_);
        
        printf("min_leaf_node %d\n", min_leaf_node_);
        printf("min_split_node %d\n", min_split_node_);
        
        printf("candidate_dim_num %d\n", candidate_dim_num_);
        printf("candidate_threshold_num %d\n", candidate_threshold_num_);
        
   //     printf("feature_dim %d\n", feature_dim_);
   //     printf("label_dim %d\n", label_dim_);
        
        printf("verbose %d\n", (int)verbose_);
        printf("verbose_leaf %d\n\n", (int)verbose_leaf_);
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
    // foldNum: 10,
    // foldIndex: 0, 1, 2, ... 9
    static void cross_validation_split(const int sampleNum, const int foldNum, const int foldIndex,
                                       vector<unsigned int> & trainingIndices, vector<unsigned> & testingIndices);
    
 //   static Eigen::VectorXd concat_vector(const vector<VectorXd> & data);
    
    
};





#endif /* defined(__Classifer_RF__DTRUtil__) */
