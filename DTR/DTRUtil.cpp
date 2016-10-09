//
//  DTRUtil.cpp
//  Classifer_RF
//
//  Created by jimmy on 2016-10-03.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#include "DTRUtil.h"
#include <algorithm>


void DTRUtil::mean_stddev(const vector<VectorXd> & labels,
                          const vector<unsigned int> & indices,
                          VectorXd & mean, VectorXd & sigma)
{
    assert(indices.size() > 0);
    
    mean = Eigen::VectorXd::Zero(labels[0].size());
    
    for (int i = 0; i<indices.size(); i++) {
        int index = indices[i];
        assert(index >= 0 && index < labels.size());
        mean += labels[index];
    }
    mean /= indices.size();
    
    sigma = Eigen::VectorXd::Zero(labels[0].size());
    if (indices.size() == 1) {
        return;
    }
    for (int i = 0; i<indices.size(); i++) {
        Eigen::VectorXd dif = labels[indices[i]] - mean;
        for (int j = 0; j<sigma.size(); j++) {
            sigma[j] += dif[j] * dif[j];
        }
    }
    for (int j = 0; j<sigma.size(); j++) {
        sigma[j] = sqrt(fabs(sigma[j]));
    }
}

double DTRUtil::spatial_variance(const vector<VectorXd> & labels,
                                 const vector<unsigned int> & indices)
{
    if (indices.size() <= 0) {
        return 0.0;
    }
    assert(indices.size() > 0);    
    
    Eigen::VectorXd mean = Eigen::VectorXd::Zero(labels[0].size());
    
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
        Eigen::VectorXd dif = labels[index] - mean;
        for (int j = 0; j<dif.size(); j++) {
            var += dif[j] * dif[j];
        }
    }
    return var;
}

void DTRUtil::mean_median_error(const vector<VectorXd> & errors,
                                Eigen::VectorXd & mean,
                                Eigen::VectorXd & median)
{
    assert(errors.size() > 0);
    const int dim = (int)errors[0].size();
    mean = Eigen::VectorXd::Zero(dim);
    median = Eigen::VectorXd::Zero(dim);
    
    vector<vector<double> > each_dim_data(dim);
    for (int i = 0; i<errors.size(); i++) {
        Eigen::VectorXd err = errors[i].cwiseAbs();
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

void DTRUtil::cross_validation_split(const int sampleNum, const int foldNum, const int foldIndex,
                                     vector<unsigned int> & trainingIndices, vector<unsigned> & testingIndices)
{
    int startIndex = sampleNum * foldIndex/foldNum;
    int stopIndex  = sampleNum * (foldIndex + 1)/foldNum;
    for (int i = 0; i<sampleNum; i++) {
        if (i >= startIndex && i < stopIndex) {
            testingIndices.push_back(i);
        }
        else {
            trainingIndices.push_back(i);
        }
    }
    assert(trainingIndices.size() + testingIndices.size() == sampleNum);
}



