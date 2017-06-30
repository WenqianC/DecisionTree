//
//  multiclass_logistic_regression.cpp
//  Classifer_RF
//
//  Created by jimmy on 2017-06-16.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "multiclass_logistic_regression.hpp"
#include <iostream>

using Eigen::VectorXf;
using Eigen::MatrixXf;
using std::cout;
using std::endl;

namespace dt {
    static VectorXf softmax(const MatrixXf& wt,
                            const VectorXf& bias,
                            const Eigen::VectorXf& x)
    {
        Eigen::VectorXf a = wt * x + bias;  // linear activation
        // prevent overflow
        a = a - a.array().maxCoeff()*Eigen::VectorXf::Ones(a.size());
        Eigen::VectorXf exp_a = a.array().exp();
        float s = exp_a.array().sum();
        assert(s != 0);
        Eigen::VectorXf probability = exp_a/s;        
        return probability;
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
    
    static double crossEntropy(const unsigned int label,
                               const Eigen::VectorXf& probability)
    {
        assert(label < probability.size());
        double cross_entropy = - (log(probability[label]));
        return cross_entropy;
    }
    
    MultiClassLogisticRegression::~MultiClassLogisticRegression()
    {
        
    }
    
    // const vector<Eigen::VectorXf> & features
    bool MultiClassLogisticRegression::fit(const vector<Eigen::VectorXf> & features,
                                           const vector<unsigned int> & labels,
                                           const int n_category,
                                           const int max_iters)
    {
        assert(features.size() == labels.size());
        const int n_dim = (int)features[0].size();
        const int n_data = (int)features.size();
        num_category_ = n_category;
        
        // initialize parameters
        weight_ = Eigen::MatrixXf::Random(n_dim, n_category);
        bias_ = Eigen::VectorXf::Random(n_category);
        
        // the convergence rate
        vector<unsigned int> predictions(n_data);
        Eigen::MatrixXf dw = Eigen::MatrixXf::Zero(n_dim, n_category);
        Eigen::VectorXf d_bias = Eigen::VectorXf::Zero(n_category);
        //const Eigen::VectorXf one = Eigen::VectorXf::Ones(n_category);
       
        double best_loss = std::numeric_limits<double>::max();
        Eigen::MatrixXf best_w = Eigen::MatrixXf::Zero(n_dim, n_category);
        Eigen::VectorXf best_bias = Eigen::VectorXf::Zero(n_category);
        for (int iter = 0; iter < max_iters; iter++) {
            dw.setZero();
            d_bias.setZero();
            
            Eigen::MatrixXf wt = weight_.transpose();  //
            double loss = 0.0f;
            for (int n = 0; n<n_data; n++) {
                Eigen::VectorXf phi = features[n];
                // step 1. prediction by previsou weights
                Eigen::VectorXf probability = softmax(wt, bias_, phi);
                assert(probability.size() == n_category);
                probability.maxCoeff(&predictions[n]);
                
                // step 2. update weights
                // M.Bishop book, page 206. 4.109
                for (int j = 0; j<n_category; j++) {
                    int t_nj = (labels[n] == j)?1:0;
                    float y_nj = probability[j];
                    dw.col(j) += (y_nj - t_nj) * phi;
                    d_bias[j] += (y_nj - t_nj)/n_category;
                }
                
                // update loss
                loss += crossEntropy(labels[n], probability);
            }
            
            loss /= n_data;
            
            //cout<<"weight update is: \n"<<dw<<endl;
            loss += 0.5 * weight_.norm();
            
            if (loss < best_loss) {
                best_loss = loss;
                best_w = weight_;
                best_bias = bias_;
            }
            
            Eigen::MatrixXf cur_w   = weight_ - learning_rate_ * dw - learning_rate_ * lambda_ * weight_;
            Eigen::VectorXf cur_bias = bias_  - learning_rate_ * d_bias;
            
            // step 3, early stop  ?
            //printf("iteration %d, accuracy %lf\n", iter, acc);
            
            // step 3
            weight_ = cur_w;
            bias_ = cur_bias;
            //cout<<"current weight is: \n"<<weight_<<endl;
        }
        weight_ = best_w;
        bias_ = best_bias;
        printf("best training loss %lf\n", best_loss);
        return true;
    }
    
    bool MultiClassLogisticRegression::predict(const vector<Eigen::VectorXf> & features,
                                               vector<unsigned int> & predictions) const
    {
        
        assert(features.size() > 0);
        //assert(features[0].size() == num_category_);
        
        Eigen::MatrixXf wt = weight_.transpose();
        assert(wt.cols() == features[0].size());
        for (int i = 0; i<features.size(); i++) {
            int pred = 0;
            Eigen::VectorXf probability = softmax(wt, bias_, features[i]);
            probability.maxCoeff(&pred);
            predictions.push_back(pred);            
        }
        assert(features.size() == predictions.size());
        return true;
    }
    
    unsigned int MultiClassLogisticRegression::predict(const Eigen::VectorXf & feature) const
    {
        Eigen::MatrixXf wt = weight_.transpose();
        assert(wt.cols() == feature.size());
        
        int pred = 0;
        Eigen::VectorXf probability = softmax(wt, bias_, feature);
        probability.maxCoeff(&pred);
        assert(pred >= 0);
        return (unsigned int)pred;
    }
    
    // after fit the model
    void MultiClassLogisticRegression::getParameter(Eigen::MatrixXf & weight, Eigen::VectorXf & bias) const
    {
        assert(bias_.size() > 0);
        assert(bias_.size() == weight_.cols());
        
        weight = weight_;
        bias = bias_;
    }
    
    void MultiClassLogisticRegression::setParameter(const Eigen::MatrixXf & weight,
                                                    const Eigen::VectorXf & bias)
    {
        weight_ = weight;
        bias_ = bias;        
    }
    
    
    
} // name spacd dt
