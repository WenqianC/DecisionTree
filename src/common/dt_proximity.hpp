//
//  dt_proximity.h
//  SequentialRandomForest
//
//  Created by jimmy on 2017-07-14.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __SequentialRandomForest__dt_proximity__
#define __SequentialRandomForest__dt_proximity__

// proximity matrix of random forest
// https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm
// brief: proximity matrix is used to measure the similarity between examples
// In deicision tree, examples in the same leaf node are considered as "similar"

#include <stdio.h>
#include <Eigen/Dense>
#include <unordered_map>

using Eigen::MatrixXf;

class DTProximity
{
public:
    DTProximity(int n)
    {
        num_ = n;
    }
    ~DTProximity(){}
    
    // add one example using index
    void addExample(const int index1, const int index2);
    
    // k: most k similar example for a particular example
    void computeProximityMatrix(const int k);
    
    const Eigen::MatrixXf & getMatrix(void) const
    {
        assert(prox_.rows() == num_);
        return prox_;
    }
    
private:
    // approximate matrix
    MatrixXf prox_;
    
    // count of numbers that two example in the same leaf node
    // the key is ordered, k.first < k.second
    std::unordered_map<int, double> count_;
    
    int num_;       // number of examples
private:
    
    int indexToKey(const int index1, const int index2) const;
    void keyToIndex(const int key, int & index1, int & index2) const;
    
    
    
};

#endif /* defined(__SequentialRandomForest__dt_proximity__) */
