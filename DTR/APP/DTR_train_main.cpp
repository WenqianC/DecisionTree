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
#include "DTCUtil.h"


static void help()
{
    printf("program    featureFile labelFile DTParamFile saveFile\n");
    printf("DTR_train  feature.txt label.txt param.txt   dt_model.txt\n");
    
}

int main(int argc, const char * argv[])
{
    if (argc != 5) {
        printf("argc is %d, should be 5.\n", argc);
        help();
        return -1;
    }
    
    const char *feature_file = argv[1];
    const char *label_file = argv[2];
    const char *param_file = argv[3];
    const char *save_file = argv[4];
    
    vector<VectorXd> features;
    vector<VectorXd> labels;
    DTCUtilIO::read_matrix(feature_file, features);
    DTCUtilIO::read_matrix(label_file, labels);
    
    assert(features.size() == labels.size());
    assert(features.size() > 0);
    
    DTRTreeParameter tree_param;
    bool is_read = tree_param.readFromFile(param_file);
    assert(is_read);
    tree_param.feature_dim_ = (int)features[0].size();
    tree_param.label_dim_ = (int)labels[0].size();
    
    DTRegressorBuilder builder;
    builder.setTreeParameter(tree_param);
    
    DTRegressor model;
    builder.buildModel(model, features, labels, save_file);
    
    model.save(save_file);
    printf("save model to %s\n", save_file);
    tree_param.printSelf();
    
    return 0;
}
#endif

