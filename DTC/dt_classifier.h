//
//  DTClassifier.h
//  Classifer_RF
//
//  Created by jimmy on 2016-09-20.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef __Classifer_RF__DTClassifier__
#define __Classifer_RF__DTClassifier__

#include <stdio.h>
#include <vector>
#include "dtc_tree.h"

using std::vector;

class DTClassifer
{
    friend class DTClassifierBuilder;
    
    vector<DTCTree* > trees_;
    DTCTreeParameter tree_param_;
    
public:
    bool predict(const Eigen::VectorXf & feature,
                 Eigen::VectorXf & prob) const;
    
    bool predict(const Eigen::VectorXf & feature,
                 int & pred);
    
    bool save(const char *fileName) const;
    bool load(const char *fileName);
};

#endif /* defined(__Classifer_RF__DTClassifier__) */
