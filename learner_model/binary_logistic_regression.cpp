//
//  binary_logistic_regression.cpp
//  Classifer_RF
//
//  Created by jimmy on 2017-06-15.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "binary_logistic_regression.h"
#include <iostream>

using std::cout;
using std::endl;

namespace dt {
    static double sigmod(double x){
        return 1.0/(1.0+exp(-x));
    }
    
    static double accuracy(const vector<unsigned int>& labels,
                           const vector<unsigned int>& predictions)
    {
        assert(labels.size() == predictions.size());
        double n_fitted = 0;
        for (int i = 0; i<labels.size(); i++) {
            if (labels[i] == predictions[i]) {
                n_fitted += 1.0;
            }
        }
        return n_fitted/labels.size();
    }
    
    static double crossEntropy(const vector<unsigned int>& labels,
                               const vector<double>& probability)
    {
        double cross_entropy = 0.0;
        for (int i = 0; i<labels.size(); i++) {
            double prob = probability[i];
            cross_entropy += - (labels[i] * log(prob) + (1.0 - labels[i])*log(1.0 - prob));
        }
        cross_entropy /= labels.size();
        return cross_entropy;
    }
    
    BinaryLogisticRegression::~BinaryLogisticRegression()
    {
        
    }
    
    // const vector<Eigen::VectorXf> & features
    bool BinaryLogisticRegression::fit(const vector<Eigen::VectorXf> & features,
                                       const vector<unsigned int> & labels,
                                       const int max_iters)
    {
        assert(features.size() == labels.size());
        const int n_dim = (int)features[0].size();
        const int n_data = (int)features.size();
        
        weight_ = Eigen::VectorXf::Random(n_dim);
        bias_ = 0.0f;
        
        // the convergence rate
        vector<double> predictions(n_data);
        Eigen::VectorXf dw = Eigen::VectorXf::Zero(n_dim);
        float d_bias = 0.0f;
        double best_loss = std::numeric_limits<double>::max();
        double pre_loss = INT_MAX;
        Eigen::VectorXf best_w = Eigen::VectorXf::Zero(n_dim);
        float best_bias = 0.0f;
        
        for (int iter = 0; iter < max_iters; iter++) {
            // step 1. prediction by previsou weights
            for (int i = 0; i<features.size(); i++) {
                double phi = features[i].dot(weight_) + bias_;
                double prob = sigmod(phi);
                predictions[i] = prob;
            }
            // step 2. update weights
            dw.setZero();
            d_bias = 0.0f;
            for (int i = 0; i<predictions.size(); i++) {
                double dif = predictions[i] - labels[i];
                if (dif != 0) {
                    // update each weight
                    dw += dif * features[i];
                    d_bias += dif * 0.5;
                }
            }
            
            double loss = crossEntropy(labels, predictions);  // data fitting term
            loss += 0.5 * lambda_ * weight_.norm();           // regularization term
            if (loss < best_loss) {
                best_loss = loss;
                best_w = weight_;
                best_bias = bias_;
            }
            
            // step 3, early stop
           // double loss_update = loss - pre_loss;
            
            Eigen::VectorXf cur_w = weight_ - learning_rate_ * dw - learning_rate_ * lambda_ * weight_;
            float cur_bias = bias_ - learning_rate_ * d_bias;
            
            
            
           // printf("iteration: %d, loss: %lf loss update: %lf \n", iter, loss, loss_update);
            
            // step 3
            weight_ = cur_w;
            bias_ = cur_bias;
            pre_loss = loss;
            
        }
        weight_ = best_w;
        bias_ = best_bias;
        //printf("training best loss %lf\n", best_loss);
        return true;
    }
    
    bool BinaryLogisticRegression::predict(const vector<Eigen::VectorXf> & features,
                                           vector<unsigned int> & predictions)
    {
        assert(features.size() > 0);
        assert(features[0].size() == weight_.size());
        
        for (int i = 0; i<features.size(); i++) {
            double phi = features[i].dot(weight_) + bias_;
            double a = sigmod(phi);
            if (a < 0.5) {
                predictions.push_back(0);
            }
            else {
                predictions.push_back(1);
            }
        }
        assert(features.size() == predictions.size());
        return true;
    }
    
}