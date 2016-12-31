//
//  DTRandom.cpp
//  Classifer_RF
//
//  Created by jimmy on 2016-09-20.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#include "DTRandom.h"
#include "vnl_random.h"
#include <cassert>

void DTRandom::outof_bag_sampling(const unsigned int N,
                                  vector<unsigned int> & bootstrapped,
                                  vector<unsigned int> & outof_bag)
{
    vnl_random rnd;
    
    vector<bool> isPicked(N, false);
    for (int i = 0; i<N; i++) {
        int idx = rnd.lrand32(0, N-1);
        bootstrapped.push_back(idx);
        isPicked[idx] = true;
    }
    
    for (int i = 0; i<N; i++) {
        if (!isPicked[i]) {
            outof_bag.push_back(i);
        }
    }
}

vector<double>
DTRandom::generateRandomNumber(const double min_v, const double max_v, int num)
{
    assert(min_v < max_v);
    
    vector<double> values;
    vnl_random rnd;
    for (int i = 0; i<num; i++) {
        double v = rnd.drand32(min_v, max_v);
        values.push_back(v);
    }
    return values;
}
