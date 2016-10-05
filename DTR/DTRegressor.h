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
    friend class DTRegressorBuilder;
    
    vector<DTRTree* > trees_;
    DTRTreeParameter reg_tree_param_;
    
public:
    
    bool predict(const Eigen::VectorXd & feature,
                 Eigen::VectorXd & pred) const;
    
    bool save(const char *fileName) const;
    bool load(const char *fileName);
};


#endif /* defined(__Classifer_RF__DTRegressor__) */
