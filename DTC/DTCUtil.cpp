//
//  DTCUtil.cpp
//  Classifer_RF
//
//  Created by jimmy on 2016-07-26.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#include "DTCUtil.h"
#include "vnl_random.h"
#include <assert.h>

vector<double>
DTCUtil::generateRandomNumber(const double min_v, const double max_v, int num)
{
    assert(min_v < max_v);
    
    vector<double> values;
    vnl_random rnd;
    for (int i = 0; i<num; i++) {
        double v = rnd.drand32(min_v, max_v);
        values.push_back(v);
    }
    return values;
}

double
DTCUtil::crossEntropy(const VectorXd & prob)
{
    double entropy = 0.0;
    for (int i = 0; i<prob.size(); i++) {
        double p = prob[i];
        assert(p > 0 && p <= 1);
        entropy += - p * std::log(p);
    }
    return entropy;
}

bool
DTCUtil::isSameLabel(const vector<unsigned int> & labels, const vector<unsigned int> & indices)
{
    assert(indices.size() >= 1);
    unsigned label = labels[indices[0]];
    for (int i = 1; i<indices.size(); i++) {
        if (label != labels[indices[i]]) {
            return false;
        }
    }
    return true;
}

Eigen::MatrixXd
DTCUtil::confusionMatrix(const vector<Eigen::VectorXd> & probs, const vector<unsigned int> & labels)
{
    assert(probs.size() == labels.size());
    assert(probs.size() > 0);
    
    const size_t dim = probs[0].size();
    
    Eigen::MatrixXd confusion = Eigen::MatrixXd::Zero(dim, dim);
    for (int i = 0; i<probs.size(); i++) {
        int pred = 0;
        int gd = labels[i];
        probs[i].maxCoeff(&pred);
        confusion(gd, pred) += 1.0;
    }
    return confusion;
}



