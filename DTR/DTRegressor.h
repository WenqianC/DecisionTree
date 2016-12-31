//
//  DTRegressor.h
//  Classifer_RF
//
//  Created by jimmy on 2016-10-04.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef __Classifer_RF__DTRegressor__
#define __Classifer_RF__DTRegressor__

#include <stdio.h>
#include <vector>
#include "DTRTree.h"

using std::vector;

class DTRegressor
{
public:
    friend class DTRegressorBuilder;
    friend class RDTBuilder;
    
    vector<DTRTree* > trees_;
    DTRTreeParameter reg_tree_param_;
    
    int feature_dim_;       // feature dimension
    int label_dim_;
    
public:
    DTRegressor(){feature_dim_ = 0; label_dim_ = 0;}
    ~DTRegressor(){}
    
    bool predict(const Eigen::VectorXd & feature,
                 Eigen::VectorXd & pred) const;
    
    // return every prediction from every tree
    bool predict(const Eigen::VectorXd & feature,
                 vector<Eigen::VectorXd> & predictions) const;
    
    bool save(const char *fileName) const;
    bool load(const char *fileName);
    
    // for debug
    
};


#endif /* defined(__Classifer_RF__DTRegressor__) */
