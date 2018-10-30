//
//  main.cpp
//  Classifer_RF
//
//  Created by jimmy on 2016-07-26.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#if 0
#include <iostream>
#include "DTRegressor.h"
#include "DTRegressorBuilder.h"
#include "dt_util_io.hpp"

using namespace std;

static void help()
{
    printf("program    modelFile featureFile labelFile saveFile\n");
    printf("DTR_train  model.txt feature.txt label.txt pred.txt\n");
}

int main(int argc, const char * argv[])
{
    if (argc != 5) {
        printf("argc is %d, should be 5.\n", argc);
        help();
        return -1;
    }
    
    const char *model_file = argv[1];
    const char *feature_file = argv[2];
    const char *label_file = argv[3];
    const char *save_file = argv[4];
    
    // load feature and label
    vector<VectorXd> features;
    vector<VectorXd> labels;
    bool is_read = DTUtil_IO::read_matrix(feature_file, features);
    assert(is_read);
    is_read = DTUtil_IO::read_matrix(label_file, labels);
    assert(is_read);
    
    assert(features.size() == labels.size());
    assert(features.size() > 0);
    
    // load model
    DTRegressor model;
    is_read = model.load(model_file);
    assert(is_read);
    
    vector<Eigen::VectorXd> predictions;    
    for (int i =0; i<features.size(); i++) {
        Eigen::VectorXd pred;
        model.predict(features[i], pred);
        predictions.push_back(pred);
    }
    
    assert(predictions.size() == labels.size());
  
    DTUtil_IO::save_matrix(save_file, predictions);
    
    // analyze prediction error
    vector<Eigen::VectorXd> errors;
    for (int i = 0; i<labels.size(); i++) {
        Eigen::VectorXd err = labels[i] - predictions[i];
        errors.push_back(err);
    }
    Eigen::VectorXd mean_err;
    Eigen::VectorXd median_err;
    DTRUtil::mean_median_error(errors, mean_err, median_err);
    cout<<"mean error: "<<mean_err<<" median error "<<median_err<<endl;
    
    return 0;
}
#endif

