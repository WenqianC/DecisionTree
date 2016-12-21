//
//  DTRandom.h
//  Classifer_RF
//
//  Created by jimmy on 2016-09-20.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef __Classifer_RF__DTRandom__
#define __Classifer_RF__DTRandom__

#include <stdio.h>
#include <vector>

using std::vector;

class DTRandom
{
public:
    // out of bagging sampling, the random number generator is related to the machine time
    static void outof_bag_sampling(const unsigned int N,
                                   vector<unsigned int> & bootstrapped,
                                   vector<unsigned int> & outof_bag);    
    
};

#endif /* defined(__Classifer_RF__DTRandom__) */
