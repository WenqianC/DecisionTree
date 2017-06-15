//
//  binary_logistic_regression.cpp
//  Classifer_RF
//
//  Created by jimmy on 2017-06-15.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "binary_logistic_regression.h"

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
        vector<unsigned int> predictions(n_data);
        Eigen::VectorXf dw = Eigen::VectorXf::Zero(n_dim);
        float d_bias = 0.0f;
        double best_acc = 0.0f;
        Eigen::VectorXf best_w = Eigen::VectorXf::Zero(n_dim);
        float best_bias = 0.0f;
        for (int iter = 0; iter < max_iters; iter++) {
            // step 1. prediction by previsou weights
            for (int i = 0; i<features.size(); i++) {
                double phi = features[i].dot(weight_) + bias_;
                double a = sigmod(phi);
                if (a < 0.5) {
                    predictions[i] = 0;
                }
                else {
                    predictions[i] = 1;
                }
            }
            // step 2. update weights
            dw.setZero();
            d_bias = 0.0f;
            for (int i = 0; i<predictions.size(); i++) {
                int dif = labels[i] - predictions[i];
                if (dif != 0) {
                    // update each weight
                    for (int j = 0; j<n_dim; j++) {
                        dw[j] += dif * features[i][j];
                    }
                    d_bias += 1.0f;
                }
            }
            
            double acc = accuracy(labels, predictions);
            if (acc > best_acc) {
                best_acc = acc;
                best_w = weight_;
                best_bias = bias_;
            }
            
            Eigen::VectorXf cur_w = weight_ + learning_rate_ * dw - learning_rate_ * lambda_ * weight_;
            float cur_bias = bias_ + learning_rate_ * d_bias;
            
            // step 3, early stop  ?
            
            //printf("iteration %d, learning rate %lf, weight update %f, accuracy %lf\n", iter, learning_rate_, dw.norm()/n_data, acc);
            
            // step 3
            weight_ = cur_w;
            bias_ = cur_bias;
        }
        weight_ = best_w;
        bias_ = best_bias;
        printf("training accuracy %lf\n", best_acc);
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