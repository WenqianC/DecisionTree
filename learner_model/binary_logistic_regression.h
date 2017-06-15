//
//  binary_logistic_regression.h
//  Classifer_RF
//
//  Created by jimmy on 2017-06-15.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __Classifer_RF__binary_logistic_regression__
#define __Classifer_RF__binary_logistic_regression__

// binary logistic regression
// PRML by M.Bishop page 206,
// http://www.hlt.utdallas.edu/~vgogate/ml/2015s/lectures/lr-nb-lec6.pdf
#include <stdio.h>
#include <Eigen/Dense>
#include <vector>

using std::vector;
namespace dt {
    class BinaryLogisticRegression
    {
    public:
        BinaryLogisticRegression()
        {
            learning_rate_ = 0.0001;
            lambda_ = 0.001;
        }
        ~BinaryLogisticRegression();
        
        // const vector<Eigen::VectorXf> & features
        bool fit(const vector<Eigen::VectorXf> & features,
                 const vector<unsigned int> & labels,
                 const int max_iters = 500);
        
        bool predict(const vector<Eigen::VectorXf> & features,
                     vector<unsigned int> & predictions);
                     
        
    private:
        double learning_rate_;
        double lambda_;          // regularization term
        
        Eigen::VectorXf weight_;
        float bias_;
    };
} // dt: decision tree


#endif /* defined(__Classifer_RF__binary_logistic_regression__) */
