//
//  bt_rnd_regressor.h
//  RGBD_RF
//
//  Created by jimmy on 2017-01-21.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __RGBD_RF__bt_rnd_regressor__
#define __RGBD_RF__bt_rnd_regressor__

#include <stdio.h>
#include <vector>
#include "bt_rnd_tree.h"

using std::vector;

class BTRNDRegressor
{

    friend class BTRNDRegressorBuilder;
    
    typedef BTRNDTree *  TreePtr;
    typedef SCRFRandomFeature Feature;
    
    vector<TreePtr> trees_;
    BTRNDTreeParameter reg_tree_param_;
    DatasetParameter dataset_param_;
    
    
    int label_dim_;
    
public:
    BTRNDRegressor(){label_dim_ = 0;}
    ~BTRNDRegressor(){}  // @todo release node
    
    // return every prediction and distance from every tree
    bool predict(const Feature & feature,
                 const cv::Mat & rgb_image,
                 const int max_check,
                 vector<Eigen::VectorXf> & predictions,
                 vector<float> & dists) const;
    
    const BTRNDTreeParameter & getTreeParameter(void) const;
    const DatasetParameter & getDatasetParameter(void) const;
    const BTRNDTree * getTree(int index) const;
    
    
    bool save(const char *fileName) const;
    bool load(const char *fileName);
    
    
};


#endif /* defined(__RGBD_RF__bt_rnd_regressor__) */
