//
//  DTUtil_IO.cpp
//  Classifer_RF
//
//  Created by jimmy on 2016-10-04.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#include "DTUtil_IO.h"

/********         DTUtil_IO           ************/
bool DTUtil_IO::read_matrix(const char * file_name, vector<VectorXd> & data)
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

bool DTUtil_IO::read_matrix(const char * file_name, Eigen::MatrixXd & data)
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
    data = Eigen::MatrixXd::Zero(rows, cols);
    for (int i = 0; i<rows; i++) {
        double val = 0;
        for (int j = 0; j<cols; j++) {
            num = fscanf(pf, "%lf", & val);
            assert(num == 1);
            data(i, j) = val;
        }
    }
    fclose(pf);
    printf("read data: %lu %lu \n", data.rows(), data.cols());
    return true;
}

bool DTUtil_IO::read_fn_matrix(const char *file_name, vector<int> & fns, vector<Eigen::VectorXd> & features)
{
    assert(fns.size() == 0);
    assert(features.size() == 0);
    
    vector<Eigen::VectorXd> fn_data;
    bool is_read = DTUtil_IO::read_matrix(file_name, fn_data);
    assert(is_read);
    assert(fn_data[0].size() > 1);
    
    int feat_size = (int)fn_data[0].size() - 1;
    for (int i = 0; i<fn_data.size(); i++) {
        // treat first column as frame number
        Eigen::VectorXd cur_feat = Eigen::VectorXd::Zero(feat_size);
        fns.push_back((int)fn_data[i][0]);
        // the rest column as feature
        for (int j = 1; j<fn_data[i].size(); j++) {
            cur_feat[j-1] = fn_data[i][j];
        }
        features.push_back(cur_feat);
    }
    assert(features.size() == fns.size());
    return true;
}

bool DTUtil_IO::read_fn_labels(const char * file_name, vector<int> & fns, vector<unsigned int> & labels)
{
    assert(fns.size() == 0);
    assert(labels.size() == 0);
    
    vector<Eigen::VectorXd> fn_data;
    bool is_read = DTUtil_IO::read_matrix(file_name, fn_data);
    assert(is_read);
    assert(fn_data[0].size() == 2);
    
    for (int i = 0; i<fn_data.size(); i++) {
        int fn = (int)fn_data[i][0];
        unsigned int label = (unsigned int)fn_data[i][1];
        fns.push_back(fn);
        labels.push_back(label);
    }
    assert(fns.size() == labels.size());
    return true;
}

bool DTUtil_IO::read_fn_gd_preds(const char *file_name, vector<int> & fns, vector<unsigned int> & gds,  vector<unsigned int> & preds)
{
    assert(fns.size() == 0);
    assert(gds.size() == 0);
    assert(preds.size() == 0);
    
    vector<Eigen::VectorXd> fn_data;
    bool is_read = DTUtil_IO::read_matrix(file_name, fn_data);
    assert(is_read);
    assert(fn_data[0].size() == 3);
    
    for (int i = 0; i<fn_data.size(); i++) {
        int fn = (int)fn_data[i][0];
        unsigned int gd = (unsigned int)fn_data[i][1];
        unsigned int pred = (unsigned int)fn_data[i][2];
        fns.push_back(fn);
        gds.push_back(gd);
        preds.push_back(pred);
    }
    assert(fns.size() == gds.size());
    assert(fns.size() == preds.size());
    return true;
}

bool DTUtil_IO::save_matrix(const char * file_name, const vector<VectorXd> & data)
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

bool DTUtil_IO::read_labels(const char * file_name, vector<unsigned int> & labels)
{
    vector<Eigen::VectorXd> data;
    bool is_read = DTUtil_IO::read_matrix(file_name, data);
    assert(is_read);
    assert(data[0].size() == 1);
    for (int i = 0; i<data.size(); i++) {
        int val = (unsigned int)data[i][0];
        labels.push_back(val);
    }
    return true;
}

bool DTUtil_IO::read_files(const char *file_name, vector<string> & files)
{
    FILE *pf = fopen(file_name, "r");
    if (!pf) {
        printf("can not load from %s\n", file_name);
        return false;
    }
    assert(pf);
    while (1) {
        char line[1024] = {NULL};
        int ret = fscanf(pf, "%s", line);
        if (ret != 1) {
            break;
        }
        files.push_back(string(line));
    }
    printf("read %lu lines\n", files.size());
    fclose(pf);
    return true;
}

