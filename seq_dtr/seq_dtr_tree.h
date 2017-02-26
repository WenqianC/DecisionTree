//
//  seq_dtr_tree.h
//  Classifer_RF
//
//  Created by jimmy on 2017-02-25.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __Classifer_RF__seq_dtr_tree__
#define __Classifer_RF__seq_dtr_tree__

// sequential decision tree (regressor) tree
#include <stdio.h>
#include "seq_dtr_util.h"
#include <Eigen/Dense>
#include <vector>
#include "DTRandom.h"

using Eigen::VectorXf;
using Eigen::MatrixXf;

using std::vector;

class SeqDTRTreeNode;

class SeqDTRTree
{
    friend class SeqDTRegressor;
private:
    typedef SeqDTRTreeNode Node;
    typedef SeqDTRTreeNode* NodePtr;
    typedef SeqDTRTreeParameter TreeParameter;
    typedef SeqDTRSplitParameter SplitParameter;
    
    typedef Eigen::MatrixXf FeatureType;
    typedef Eigen::MatrixXf LabelType;    

    NodePtr root_;
    TreeParameter tree_param_;
    
    vector<unsigned int> time_steps_; // time steps from small to large, fixed during training
    vector<double> weights_;          // weight of each time step
    DTRandom rnd_generator_;
    
public:
    SeqDTRTree(){root_ = NULL;}
    ~SeqDTRTree();
    
    // features:   matrix, each row is a vector, multiple rows are from multiple input (e.g., cameras at the same time)
    // label_seqs: a sequence of labels
    // indices:
    bool buildTree(const vector<MatrixXf> & features,
                   const vector<MatrixXf> & label_seqs,
                   const vector<unsigned int> & indices,
                   const TreeParameter & param,
                   const vector<unsigned int>& time_steps);
    
    // predictions: each row is a predicted vector at eath time step
    // raw prediction without weight
    bool rawPredict(const Eigen::MatrixXf & feature_seq,
                 vector<unsigned int> & time_steps,
                 vector<Eigen::VectorXf> & predictions) const;
    
    // assume has weight
    bool predict(const Eigen::MatrixXf & feature_seq,
                 vector<unsigned int> & time_steps,
                 vector<double> & weights,
                 vector<Eigen::VectorXf> & predictions) const;
    
    
    const TreeParameter & getTreeParameter(void) const;
    void setTreeParameter(const TreeParameter & param);
    void setTimesteps(const vector<unsigned int>& steps);
    // weight from out of bag cross validation
    bool setWeights(const vector<double> & wts);
    
private:
    
    bool configureNode(const vector<MatrixXf> & features,
                       const vector<MatrixXf> & label_seqs,
                       const vector<unsigned int> & indices,
                       NodePtr node);
    
    bool setLeafNode(const vector<Eigen::MatrixXf> & labels,
                     const vector<unsigned int> & indices,
                     NodePtr node);
    
    bool setInternalNode(const vector<Eigen::MatrixXf> & labels,
                         const vector<unsigned int> & indices,
                         NodePtr node,
                         const SplitParameter & split_param);
    
    bool bestSplitParameter(const vector<Eigen::MatrixXf> & features,
                            const vector<Eigen::MatrixXf> & labels,
                            const vector<unsigned int> & indices,
                            const TreeParameter & tree_param,
                            const int depth,
                            SplitParameter & split_param,
                            vector<unsigned int> & left_indices,
                            vector<unsigned int> & right_indices);
    
    bool rawPredict(const NodePtr node,
                    const Eigen::MatrixXf & feature,
                    vector<unsigned int> & time_steps,
                    vector<Eigen::VectorXf> & predictions) const;
    
    bool predict(const NodePtr node,
                 const Eigen::MatrixXf & feature_seq,
                 vector<Eigen::VectorXf> & predictions) const;
    
    
};


#endif /* defined(__Classifer_RF__seq_dtr_tree__) */
