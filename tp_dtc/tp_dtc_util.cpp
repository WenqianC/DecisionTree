//
//  tp_dtc_util.cpp
//  Classifer_RF
//
//  Created by jimmy on 2017-02-19.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "tp_dtc_util.h"

vector< vector<int> > TPDTCUtil::generatePermutation(const vector<int> input_data)
{
    vector<int> data = input_data;
    vector<vector<int> > result;
    std::sort(data.begin(), data.end());
    do{
        result.push_back(data);
        
    } while (std::next_permutation(data.begin(), data.end()));
    
    return result;
}
