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



/********         DTCUtilIO           ************/
bool DTCUtilIO::read_matrix(const char * file_name, vector<VectorXd> & data)
{
    FILE *pf = fopen(file_name, "r");
    if (!pf) {
        printf("can not load from %s\n", file_name);
        return false;
    }
    assert(pf);
    int rows = 0;
    int cols = 0;
    int num = fscanf(pf, "%d %d", &rows, &cols);
    assert(num == 2);
    for (int i = 0; i<rows; i++) {
        VectorXd feat = VectorXd::Zero(cols);
        double val = 0;
        for (int j = 0; j<cols; j++) {
            num = fscanf(pf, "%lf", & val);
            assert(num == 1);
            feat[j] = val;
        }
        data.push_back(feat);
    }
    fclose(pf);
    printf("read data: %lu %lu \n", data.size(), data[0].size());
    return true;
}

bool DTCUtilIO::save_matrix(const char * file_name, const vector<VectorXd> & data)
{
    assert(data.size() > 0);
    FILE *pf = fopen(file_name, "w");
    if (!pf) {
        printf("can not write to %s\n", file_name);
        return false;
    }
    assert(pf);
    fprintf(pf, "%d %d\n", (int)data.size(), (int)data[0].size());
    for (int i = 0; i<data.size(); i++) {
        for (int j = 0; j<data[i].size(); j++) {
            fprintf(pf, "%lf ", data[i][j]);
            if (j == data[i].size()-1) {
                fprintf(pf, "\n");
            }
        }
    }
    printf("save to %s\n", file_name);
    return true;
}

bool DTCUtilIO::read_labels(const char * file_name, vector<unsigned int> & labels)
{
    vector<Eigen::VectorXd> data;
    bool is_read = DTCUtilIO::read_matrix(file_name, data);
    assert(is_read);
    assert(data[0].size() == 1);
    for (int i = 0; i<data.size(); i++) {
        int val = (unsigned int)data[i][0];
        labels.push_back(val);
    }
    return true;
}
