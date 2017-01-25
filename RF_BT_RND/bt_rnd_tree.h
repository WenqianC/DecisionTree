//
//  bt_rnd_tree.h
//  RGBD_RF
//
//  Created by jimmy on 2017-01-18.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __RGBD_RF__bt_rnd_tree__
#define __RGBD_RF__bt_rnd_tree__

// back tracking random tree
#include <stdio.h>
#include <Eigen/Dense>
#include <algorithm>
#include <flann/util/heap.h>
#include <flann/util/result_set.h>
#include <flann/flann.hpp>
#include "bt_rnd_util.h"
#include "DTRandom.h"


using std::vector;
using Eigen::VectorXf;
using flann::BranchStruct;

class BTRNDTreeNode;

class BTRNDTree
{
private:
    friend class BTRNDRegressor;
    
    typedef flann::L2<float> Distance;
    typedef Distance::ResultType DistanceType;
    typedef Distance::ElementType ElementType;
    
    typedef BTRNDTreeNode Node;
    typedef BTRNDTreeNode* NodePtr;
    typedef BranchStruct<NodePtr, DistanceType > BranchSt;
    typedef BranchSt* Branch;
    
    typedef SCRFRandomFeature FeatureType;

public: // for test only
    
    NodePtr root_;
    BTRNDTreeParameter tree_param_;
    
    Distance distance_;   // the distance functor
    int leaf_node_num_;   // total leaf node number
    vector<NodePtr> leaf_nodes_;   // leaf node for back tracking
    
    DTRandom rnd_generator_;
    
public:
    BTRNDTree();
    ~BTRNDTree();
    
    // training random forest by build a decision tree
    // samples: sampled image pixel locations
    // indices: index of samples
    // rgb_images: same size, rgb, 8bit image
    bool buildTree(const vector<FeatureType> & features,
                   const vector<VectorXf> & labels,
                   const vector<unsigned int> & indices,
                   const vector<cv::Mat> & rgb_images,
                   const BTRNDTreeParameter & param);
    
    // backtracking prediction
    bool predict(const FeatureType & feature,
                 const cv::Mat & rgb_image,
                 const int maxCheck,
                 VectorXf & pred,
                 float & dist) const;
    
    
    // each row is a descriptor
    void getLeafNodeDescriptor(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & data);
    void setLeafNodeDescriptor(const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & data);
    
    const BTRNDTreeParameter & getTreeParameter(void) const {return tree_param_;}
    void setTreeParameter(const BTRNDTreeParameter & param) {tree_param_ = param;}
    
private:
    bool configureNode(const vector<FeatureType> & features,
                       const vector<VectorXf> & labels,
                       const vector<cv::Mat> & rgb_images,
                       const vector<unsigned int> & indices,                       
                       NodePtr node);
    
    // optimize random feaute split parameter
    double optimizeRandomFeature(const vector<FeatureType> & features,
                                 const vector<VectorXf> & labels,
                                 const vector<cv::Mat> & rgbImages,
                                 const vector<unsigned int> & indices,
                                 const BTRNDTreeParameter & tree_param,
                                 const int depth,
                                 vector<unsigned int> & left_indices,
                                 vector<unsigned int> & right_indices,
                                 RandomSplitParameter & split_param);
    
    
    double bestSplitRandomParameter(const vector<FeatureType> & features,
                                    const vector<VectorXf> & labels,
                                        const vector<cv::Mat> & rgbImages,
                                        const vector<unsigned int> & indices,
                                        const BTRNDTreeParameter & tree_param,
                                    const int depth,
                                        RandomSplitParameter & split_param,
                                        vector<unsigned int> & left_indices,
                                    vector<unsigned int> & right_indices);
    
    bool setLeafNode(const vector<FeatureType> & features,
                     const vector<VectorXf> & labels,
                                const vector<unsigned int> & indices,
                                         NodePtr node);
    
    // record leaf node in an array for O(1) access
    void hashLeafNode();
    
    void recordLeafNodes(NodePtr node, vector<NodePtr> & leafNodes, int & index);
    
    
    void searchLevel(flann::ResultSet<DistanceType>  & result_set, const ElementType* vec, const NodePtr node,
                     int & check_count, const int max_check,
                     flann::Heap<BranchSt>* heap, flann::DynamicBitset& checked,
                     const FeatureType & sample,     // new added parameter
                     const cv::Mat & rgb_image) const;
    
    
    static double computeRandomFeature(const cv::Mat & rgb_image, const FeatureType * feat, const RandomSplitParameter & split);    
    
    
};


#endif /* defined(__RGBD_RF__bt_rnd_tree__) */
