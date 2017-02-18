//
//  dt_util.cpp
//  Classifer_RF
//
//  Created by jimmy on 2017-02-16.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "dt_util.h"

template <class T>
double DTUtil::spatialVariance(const vector<T> & labels, const vector<unsigned int> & indices)
{
    if (indices.size() <= 0) {
        return 0.0;
    }
    assert(indices.size() > 0);
    
    T mean = T::Zero(labels[0].size());
    
    for (int i = 0; i<indices.size(); i++) {
        int index = indices[i];
        assert(index >= 0 && index < labels.size());
        mean += labels[index];
    }
    mean /= indices.size();
    
    double var = 0.0;
    for (int i = 0; i<indices.size(); i++) {
        int index = indices[i];
        assert(index >= 0 && index < labels.size());
        T dif = labels[index] - mean;
        for (int j = 0; j<dif.size(); j++) {
            var += dif[j] * dif[j];
        }
    }
    return var;
}

template<class Type1, class Type2>
double DTUtil::spatialVariance(const vector<Type1> & labels, const vector<unsigned int> & indices, const vector<Type2> & wt)
{
    if (indices.size() <= 0) {
        return 0.0;
    }
    assert(indices.size() > 0);
    assert(wt.size() == labels.front().size());
    
    Type1 mean = Type1::Zero(labels[0].size());
    
    for (int i = 0; i<indices.size(); i++) {
        int index = indices[i];
        assert(index >= 0 && index < labels.size());
        mean += labels[index];
    }
    mean /= indices.size();
    
    double var = 0.0;
    for (int i = 0; i<indices.size(); i++) {
        int index = indices[i];
        assert(index >= 0 && index < labels.size());
        Type1 dif = labels[index] - mean;
        for (int j = 0; j<dif.size(); j++) {
            var += dif[j] * dif[j] * fabs(double(wt[j]));
        }
    }
    return var;
}

template <class T>
void DTUtil::meanStddev(const vector<T> & labels, const vector<unsigned int> & indices, T & mean, T & sigma)
{
    assert(indices.size() > 0);
    
    mean = T::Zero(labels[0].size());
    
    for (int i = 0; i<indices.size(); i++) {
        int index = indices[i];
        assert(index >= 0 && index < labels.size());
        mean += labels[index];
    }
    mean /= indices.size();
    
    sigma = T::Zero(labels[0].size());
    if (indices.size() == 1) {
        return;
    }
    for (int i = 0; i<indices.size(); i++) {
        T dif = labels[indices[i]] - mean;
        for (int j = 0; j<sigma.size(); j++) {
            sigma[j] += dif[j] * dif[j];
        }
    }
    for (int j = 0; j<sigma.size(); j++) {
        sigma[j] = sqrt(fabs(sigma[j])/indices.size());
    }
}

template <class T>
void DTUtil::meanMedianError(const vector<T> & errors,
                                  T & mean,
                                  T & median)
{
    assert(errors.size() > 0);
    const int dim = (int)errors[0].size();
    mean = T::Zero(dim);
    median = T::Zero(dim);
    
    vector<vector<double> > each_dim_data(dim);
    for (int i = 0; i<errors.size(); i++) {
        T err = errors[i].cwiseAbs();
        mean += err;
        for (int j = 0; j<err.size(); j++) {
            each_dim_data[j].push_back(err[j]);
        }
    }
    mean /= errors.size();
    
    for (int i = 0; i<each_dim_data.size(); i++) {
        std::sort(each_dim_data[i].begin(), each_dim_data[i].end());
        median[i] = each_dim_data[i][each_dim_data[i].size()/2];
    }
}


template double
DTUtil::spatialVariance(const vector<Eigen::VectorXf> & labels, const vector<unsigned int> & indices);

template double
DTUtil::spatialVariance(const vector<Eigen::VectorXf> & labels, const vector<unsigned int> & indices, const vector<int> & wt);

template void
DTUtil::meanStddev(const vector<Eigen::VectorXf> & labels, const vector<unsigned int> & indices, Eigen::VectorXf & mean, Eigen::VectorXf & sigma);

template void
DTUtil::meanMedianError(const vector<Eigen::VectorXf> & errors, Eigen::VectorXf & mean, Eigen::VectorXf & median);

