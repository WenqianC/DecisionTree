//  Created by jimmy on 2016-12-30.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef __BT_DT_Regressor__
#define __BT_DT_Regressor__

#include <stdio.h>
#include <vector>
#include "bt_dtr_tree.h"

using std::vector;

class BTDTRegressor
{
public:
    friend class BTDTRegressorBuilder;
    
    vector<BTDTRTree* > trees_;
    BTDTRTreeParameter reg_tree_param_;
    
    int feature_dim_;       // feature dimension
    int label_dim_;
    
public:
    BTDTRegressor(){feature_dim_ = 0; label_dim_ = 0;}
    ~BTDTRegressor(){}    
    
    bool predict(const Eigen::VectorXf & feature,
                 const int maxCheck,
                 Eigen::VectorXf & pred) const;
    
    // return every prediction from every tree
    bool predict(const Eigen::VectorXf & feature,
                 const int maxCheck,
                 vector<Eigen::VectorXf> & predictions) const;
    
    // return every prediction and distance from every tree
    bool predict(const Eigen::VectorXf & feature,
                 const int maxCheck,
                 vector<Eigen::VectorXf> & predictions,
                 vector<float> & dists) const;
    
    
    bool saveModel(const char *file_name) const;
    bool load(const char *file_name);
    
};


#endif /* defined(__RGBD_RF__BTDTRegressor__) */
