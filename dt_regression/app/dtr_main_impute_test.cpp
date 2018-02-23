//
//  dtr_main_imputation.cpp
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
    printf("program          testXFile  testYFile  useTestY modelFilePrefix  resultFile \n");
    printf("DTR_impute_test  test_x.mat test_y.mat 1        model.txt        imputed_feature.mat \n");
    printf("testXFile: testing feature file. 'feature' \n");
    printf("testYFile: testing label file. 'label' optional  \n");
    printf("useTestY: 0 --> no testYFile, 1 --> has testYFile \n");
    printf("resultFile: .mat file, has 'imputed_feature'. \n");
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

static void readRegressionFeature(const char *feature_file,
                                  vector<Eigen::VectorXf> & features)
{
    Eigen::MatrixXf mat_features;
    Eigen::MatrixXf mat_labels;
    matio::readMatrix(feature_file, "feature", mat_features);
    
    for (int i = 0; i < mat_features.rows(); i++) {
        Eigen::VectorXf feat = mat_features.row(i);
        features.push_back(feat);
        
    }
    printf("read %lu feature examples\n", features.size());
}

static vector<string> redSubmodelFiles(const char *model_file)
{
    vector<string> sub_model_files;
    
    FILE *pf = fopen(model_file, "r");
    assert(pf);
    int dims = 0;
    int num = fscanf(pf, "%d", &dims);
    assert(num == 1);
    for (int i = 0; i<dims; i++) {
        int dummy = 0;
        char buf[1024] = {NULL};
        num = fscanf(pf, "%d %s", &dummy, buf);
        assert(num == 2);
        sub_model_files.push_back(string(buf));
    }
    fclose(pf);
    assert(sub_model_files.size() == dims);
    return sub_model_files;
}




int main(int argc, const char * argv[])
{
    if (argc != 6) {
        printf("argc is %d, should be 6 .\n", argc);
        help();
        return -1;
    }
   
    const char *test_feature_file = argv[1];
    const char *test_label_file = argv[2];
    const int has_testing_label = (int)strtod(argv[3], NULL);
    const char *model_file = argv[4];
    const char *save_file = argv[5];
    
    assert(has_testing_label == 1 || has_testing_label == 0);
    
    /*
    const char *train_feature_file = "/Users/jimmy/Desktop/dtr_imputation/data/train_x.mat";
    const char *train_label_file   = "/Users/jimmy/Desktop/dtr_imputation/data/train_y.mat";
    const char *test_feature_file = "/Users/jimmy/Desktop/dtr_imputation/data/test_x.mat";
    const char *test_label_file = "/Users/jimmy/Desktop/dtr_imputation/data/test_y.mat";
    const char *param_file = "/Users/jimmy/Desktop/dtr_imputation/dtr_tree_param.txt";
    const char *model_file = "debug.txt";
    const char *save_file = "result.mat";
     */
    
    // read data
    vector<Eigen::VectorXf> test_features;
    vector<Eigen::VectorXf> test_labels;
    if (has_testing_label) {
        readRegressionDataset(test_feature_file, test_label_file, test_features, test_labels);
    }
    else {
        readRegressionFeature(test_feature_file, test_features);
    }
    assert(test_features.size() != 0);
    
    // read model
    vector<string> sub_model_files = redSubmodelFiles(model_file);
    const int dims = (int)sub_model_files.size();
    printf("imputing %d dimension features \n", dims);
    
    // final result
    Eigen::MatrixXf predictions(test_features.size(), dims);
    Eigen::VectorXf mean_error(dims);    // for debug
    // treat each dimension as a regression problem;
    for (int d = 0; d<dims; d++) {
        DTRegressor model;
        model.load(sub_model_files[d].c_str());
        
        // predict on one dimension independently
        double error = 0;
        for (int i = 0; i<test_features.size(); i++) {
            Eigen::VectorXf pred;
            bool is_pred = model.predict(test_features[i], pred);
            assert(is_pred);
            assert(pred.size() == 1);
            predictions(i, d) = pred[0];
            if (has_testing_label) {
                double dif = pred[0] - test_labels[i][d];
                error += dif * dif;
            }
        }
        error = sqrt(error/test_features.size());
        mean_error[d] = error;
    }
    if (has_testing_label) {
        cout<<"L2 norm error in each dimension is "<<mean_error.transpose()<<endl;
    }
    
    // save result
    matio::writeMatrix<Eigen::MatrixXf>(save_file, "imputed_feature", predictions);
    
    return 0;
}
#endif

