//
//  mat_io.h
//  SequentialRandomForest
//
//  Created by jimmy on 2017-07-01.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __SequentialRandomForest__mat_io__
#define __SequentialRandomForest__mat_io__

// matlab input output
#include <stdio.h>
#include <Eigen/Dense>


namespace matio {
    
    template<class matrixT>
    bool readMatrix(const char *file_name, const char *var_name, matrixT & data);
    
} // name space matio



#endif /* defined(__SequentialRandomForest__mat_io__) */
