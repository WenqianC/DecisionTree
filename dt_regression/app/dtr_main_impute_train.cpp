//
//  dtr_main_impute_train.cpp
//  SequentialRandomForest
//
//  Created by jimmy on 2017-11-02.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//


#if 1
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
    printf("program      trainXFile  trainYFile  DTParamFile   modelFile  \n");
    printf("DTR_impute   train_x.mat train_y.mat dtr_param.txt model.txt \n");
    printf("trainXFile: training feature file. .mat 'feature' \n");
    printf("trainYFile: training label file.   .mat 'label' \n");
    
    printf("DTParamFile: regression tree parameter \n");
    printf("model.txt: .txt, a list of sub-models.\n");
    
}

static vector<Eigen::VectorXf> getOneDimension(const vector<Eigen::VectorXf> & data, const int dim)
{
    assert(dim >= 0 && dim < data[0].size());
    
    vector<Eigen::VectorXf> single_dim_data;
    VectorXf temp_data(1);
    for (int i = 0; i<data.size(); i++) {
        temp_data[0] = data[i][dim];
        single_dim_data.push_back(temp_data);
    }
    return single_dim_data;
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

static vector<string> generateSubmodelFiles(const char *model_file, const int dims)
{
    vector<string> sub_model_files;
    string base_name = string(model_file);
    base_name = base_name.substr(0, base_name.size()-4);
    
    FILE *pf = fopen(model_file, "w");
    fprintf(pf, "%d\n", dims);
    for (int i = 0; i<dims; i++) {
        char buf[1024] = {NULL};
        sprintf(buf, "_%03d", i);
        string file_name = base_name + string(buf) + string(".txt");
        fprintf(pf, "%d\t %s\n", i, file_name.c_str());
        sub_model_files.push_back(file_name);
    }
    fclose(pf);
    printf("generate %lu sub model files\n", sub_model_files.size());
    return sub_model_files;
}

int main(int argc, const char * argv[])
{
    if (argc != 5) {
        printf("argc is %d, should be 9 .\n", argc);
        help();
        return -1;
    }
    
    const char *train_feature_file = argv[1];
    const char *train_label_file   = argv[2];
    const char *param_file = argv[3];
    const char *model_file = argv[4];
    
    /*
    const char *train_feature_file = "/Users/jimmy/Desktop/dtr_imputation/data/train_x.mat";
    const char *train_label_file   = "/Users/jimmy/Desktop/dtr_imputation/data/train_y.mat";
    const char *param_file = "/Users/jimmy/Desktop/dtr_imputation/dtr_impute_tree_param.txt";
    const char *model_file = "impute_debug.txt";
     */
    
    vector<Eigen::VectorXf> train_features;
    vector<Eigen::VectorXf> train_labels;
    readRegressionDataset(train_feature_file, train_label_file, train_features, train_labels);
    
    const int dims = (int)train_labels[0].size();
    printf("feature, lable dimensions: %lu %lu \n", train_features[0].size(), train_labels[0].size());
    
    DTRTreeParameter param;
    param.readFromFile(param_file);
    param.feature_dimension_ = (int)train_features[0].size();
    cout<<"decition tree parameter "<<param<<endl;
    
    // prepare sub-model files
    vector<string> sub_model_files = generateSubmodelFiles(model_file, dims);
    // treat each dimension as a regression problem;
    for (int d = 0; d<dims; d++) {
        // get training label
        vector<Eigen::VectorXf> cur_train_labels = getOneDimension(train_labels, d);
        
        DTRegressor model;
        DTRegressorBuilder builder;
        builder.setTreeParameter(param);
        
        builder.buildModel(model, train_features, cur_train_labels,
                           vector<Eigen::VectorXf>(),  vector<Eigen::VectorXf>(), sub_model_files[d].c_str());
        model.save(sub_model_files[d].c_str());
    }
    
    
    return 0;
}
#endif

