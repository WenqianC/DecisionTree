//
//  bt_dtc_tree.h
//  PTZBTRF
//
//  Created by jimmy on 2017-10-13.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __PTZBTRF__bt_dtc_tree__
#define __PTZBTRF__bt_dtc_tree__

// back tracking decision tree for classification
#include <stdio.h>
#include "bt_dtc_param.h"
#include "dt_random.hpp"

// flann
#include "flann/util/heap.h"
#include "flann/util/result_set.h"
#include <flann/flann.hpp>

using flann::BranchStruct;

class BTDTCTree
{
    friend class BTDTClassifier;
    
    // decision tree parameter
    typedef BTDTCSplitParameter SplitParameter;
    typedef BTDTCTreeParameter  TreeParameter;
    
    // internal data structures
    struct Node
    {
        Node* left_child_;
        Node* right_child_;
        int depth_;         // tree depth
        bool is_leaf_;      // leaf or non-leaf node
        
        // non-leaf node parameter
        SplitParameter split_param_;  // split parameter
        
        // leaf node parameter
        VectorXf prob_;  // label probability, leaf node
        VectorXf feat_mean_; // mean value of local descriptor, e.g., SIFT
        // @tod, feature variance (diagonal)
        int index_;          // node index, for save/store tree
        
        // non-leaf node auxiliary data
        int sample_num_;              // num of training examples
        double sample_percentage_;    // sample percentage of parent node
        
        Node(int depth) {
            left_child_ = NULL;
            right_child_ = NULL;
            depth_ = depth;
            is_leaf_ = false;
            index_ = -1;
            
            sample_num_ = 0;
            sample_percentage_ = 0.0;
        }
        ~Node() {
            if (left_child_) {
                delete left_child_;
                left_child_ = NULL;
            }
            if (right_child_) {
                delete right_child_;
                right_child_ = NULL;
            }
        }
    };
    
    typedef Node* NodePtr;
    
    // back-tracking related
    typedef flann::L2<float> Distance;
    typedef Distance::ResultType DistanceType;
    typedef Distance::ElementType ElementType;
    typedef BranchStruct<NodePtr, DistanceType > BranchSt;
    typedef BranchSt* Branch;
    
    // tree structure
    NodePtr root_;
    TreeParameter tree_param_;
    DTRandom rnd_generator_;
    
    Distance distance_;   // the distance functor
    vector<NodePtr> leaf_nodes_; // leaf node for back tracking
    int leaf_node_num_;   // total leaf node number
    
public:
    BTDTCTree();
    ~BTDTCTree();
    
    // features:
    // labels: classification label
    // indices:
    bool buildTree(const vector<VectorXf> & features,
                   const vector<int> & labels,
                   const vector<int> & indices,
                   const TreeParameter & param);
    
    // prob: predicted label distribution
    // distance: feature distance
    bool predict(const Eigen::VectorXf & feature,
                 const int max_check,
                 VectorXf & prob,
                 float & distance);
    
    bool predict(const Eigen::VectorXf & feature,
                 const int max_check,
                 int & pred,
                 float & distance);
    
    
    
    // each row is a descriptor
    void getLeafNodeDescriptor(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & data);
    void setLeafNodeDescriptor(const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & data);
    
    const BTDTCTree::TreeParameter & getTreeParameter(void) const;
    void setTreeParameter(const TreeParameter & param);
    
private:
    bool buildTreeImpl(const vector<VectorXf> & features,
                       const vector<int> & labels,
                       const vector<int> & indices,
                       NodePtr node);
    
    // optimize split parameters for particular dimension
    // O(N * m): m is number of random tries.
    bool bestSplitParameter(const vector<VectorXf> & features,
                            const vector<int> & labels,
                            const vector<int> & indices,
                            const int depth,
                            SplitParameter & split_param,
                            vector<int> & left_indices,
                            vector<int> & right_indices);
    
    
    
    // set leaf node
    void setLeafNode(const vector<VectorXf> & features,
                     const vector<int> & labels,
                     const vector<int> & indices,
                     NodePtr node);
    
    // record leaf node in an array for O(1) access
    void hashLeafNode();
    
    void recordLeafNodes(const NodePtr node, vector<NodePtr> & leaf_nodes, int & leaf_node_index);
    
    
    // searchLevel from flann kd tree
    void searchLevel(flann::ResultSet<DistanceType>  & result_set, const ElementType* vec, NodePtr node,
                     const DistanceType min_dist, int & check_count, const int max_check, const float eps_error,
                     flann::Heap<BranchSt>* heap, flann::DynamicBitset& checked) const;
    
    
public:
    // read/write
    bool writeTree(const char *fileName) const;
    bool readTree(const char *fileName);
   
    static void writeNode(FILE *pf, const NodePtr node);
    static void readNode(FILE *pf, NodePtr & node, const int category_num);
};

#endif /* defined(__PTZBTRF__bt_dtc_tree__) */
