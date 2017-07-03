//
//  multiclass_logistic_regression.h
//  Classifer_RF
//
//  Created by jimmy on 2017-06-16.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __Classifer_RF__multiclass_logistic_regression__
#define __Classifer_RF__multiclass_logistic_regression__

#include <stdio.h>
#include <Eigen/Dense>
#include <vector>

using std::vector;
namespace dt {
    class MultiClassLogisticRegression
    {
    public:
        MultiClassLogisticRegression()
        {
            learning_rate_ = 0.0001;
            lambda_ = 0.0001;
        }
        ~MultiClassLogisticRegression();
        
        
        bool fit(const vector<Eigen::VectorXf> & features,
                 const vector<unsigned int> & labels,
                 const int n_category,
                 const int max_iters = 500);
        
        bool predict(const vector<Eigen::VectorXf> & features,
                     vector<unsigned int> & predictions) const;
        
        unsigned int predict(const Eigen::VectorXf & feature) const;
        
        // after fit the model
        void getParameter(Eigen::MatrixXf & weight, Eigen::VectorXf & bias) const;
        
        void setParameter(const Eigen::MatrixXf & weight,
                          const Eigen::VectorXf & bias);
        
        
    private:
        double learning_rate_;
        double lambda_;          // regularization term
        int num_category_;
        
        Eigen::MatrixXf weight_;   // m x n where m is feature dimension and n is the number of categories
        Eigen::VectorXf bias_;     // n
    };
} // dt: decision tree


#endif /* defined(__Classifer_RF__multiclass_logistic_regression__) */
