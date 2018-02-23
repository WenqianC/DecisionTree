//
//  dtr_train.cpp
//  SequentialRandomForest
//
//  Created by jimmy on 2017-11-02.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#if 0
#include <stdio.h>
#include "dt_util_io.hpp"
#include "mat_io.hpp"
#include "dt_util.hpp"
#include "dt_regressor_builder.h"
#include "dt_regressor.h"
#include <iostream>
#include <vector>

using std::cout;
using std::endl;
using std::vector;
using Eigen::VectorXf;

static void help()
{
    printf("program     trainXFile  trainYFile  DTParamFile   modelFile  \n");
    printf("DTR_train   train_x.mat train_y.mat dtr_param.txt model.txt \n");
    printf("trainXFile: training feature file. .mat 'feature' \n");
    printf("trainYFile: training label file.   .mat 'label' \n");
    
    printf("DTParamFile: regression tree parameter \n");
    printf("model.txt: .txt\n");
}

static void readRegressionDataset(const char *feature_file,
                                  const char *label_file,
                                  vector<Eigen::VectorXf> & features,
                                  vector<Eigen::VectorXf> & labels)
{
    Eigen::MatrixXf mat_features;
    Eigen::MatrixXf mat_labels;
    matio::readMatrix(feature_file, "feature", mat_features);
    matio::readMatrix(label_file, "label", mat_labels);
    assert(mat_features.rows() == mat_labels.rows());
    
    for (int i = 0; i < mat_features.rows(); i++) {
        Eigen::VectorXf feat = mat_features.row(i);
        Eigen::VectorXf label = mat_labels.row(i);
        
        features.push_back(feat);
        labels.push_back(label);
    }
    assert(features.size() == labels.size());
    printf("read %lu train examples\n", features.size());
}


int main(int argc, const char * argv[])
{
    
    if (argc != 5) {
        printf("argc is %d, should be 5 .\n", argc);
        help();
        return -1;
    }
    const char *train_feature_file = argv[1];
    const char *train_label_file   = argv[2];
    const char *param_file = argv[3];
    const char *model_file = argv[4];
    
    /*
    const char *train_feature_file = "/Users/jimmy/Desktop/cvs_eccv2018_code/data/feature/left_train_fn_feature.mat";
    const char *train_label_file   = "/Users/jimmy/Desktop/cvs_eccv2018_code/data/left_train_fn_vq.mat";
    const char *param_file = "/Users/jimmy/Desktop/dtr_param.txt";
    const char *model_file = "debug.txt";
     */
    
    
    vector<Eigen::VectorXf> train_features;
    vector<Eigen::VectorXf> train_labels;
    readRegressionDataset(train_feature_file, train_label_file, train_features, train_labels);
    
    printf("feature, lable dimensions: %lu %lu \n", train_features[0].size(), train_labels[0].size());
    
    DTRTreeParameter param;
    param.readFromFile(param_file);
    param.feature_dimension_ = (int)train_features[0].size();
    cout<<"decition tree parameter "<<param<<endl;
    
    DTRegressor model;
    DTRegressorBuilder builder;
    builder.setTreeParameter(param);
    
    builder.buildModel(model, train_features, train_labels,
                       vector<Eigen::VectorXf>(),  vector<Eigen::VectorXf>(), model_file);
    model.save(model_file);

    
    
    return 0;
}
#endif

