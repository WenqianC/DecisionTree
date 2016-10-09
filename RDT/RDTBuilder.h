//
//  RDTBuilder.h
//  Classifer_RF
//
//  Created by jimmy on 2016-10-07.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef __Classifer_RF__RDTBuilder__
#define __Classifer_RF__RDTBuilder__

// recurrent decision tree

#include <stdio.h>
#include "DTRegressor.h"
#include <vector>
#include "SeqFeatGenerator.h"
#include <string>

using std::string;
using std::vector;

struct RDTParameter
{
    int iter_num_;
    double decrease_ratio_;
    string model_file_;      // model file name
    
    RDTParameter()
    {
        iter_num_ = 2;
        decrease_ratio_ = 0.9;
    }
};

class RDTBuilder
{
public:
    RDTBuilder();
    ~RDTBuilder();
    
    bool buildModel(DTRegressor& model,
                    const vector<int> & fns,
                    const vector< Eigen::VectorXd > & inputs,
                    const vector< Eigen::VectorXd > & outputs,
                    const SeqFeatGenerator & feature_generator,
                    const RDTParameter & rdt_param,
                    const DTRTreeParameter & tree_param) const;
    
    
};

#endif /* defined(__Classifer_RF__RDTBuilder__) */
