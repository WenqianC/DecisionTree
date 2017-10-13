//
//  seq_dtc_train_valid.cpp
//  SequentialRandomForest
//
//  Created by jimmy on 2017-07-07.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#if 0

#include <stdio.h>
#include <stdio.h>
#include "dt_util_io.hpp"
#include "dt_classifier.h"
#include "dt_classifier_builder.h"
#include "mat_io.hpp"
#include "dt_util.hpp"
#include <iostream>

using std::cout;
using std::endl;

static void help()
{
    printf("program      trainXFile trainYFile validXFile validYFile useValidData DTParamFile   saveModelFile resultFile \n");
    printf("DTCTrainVali trainX.mat trainY.mat validX.mat validY.mat 0            dtc_param.txt dtc_model.txt result.mat \n");
    printf("trainXFile: training feature file.\n");
    printf("trainYFile: training label file.\n");
    printf("feature file: fn1 feat1 fn2 feat2 ... in a single file \n");
    printf("label   file: fn1 label1 fn2 label2 ... in a single file \n");
    printf("Feature file has two variables: fn, feature \n");
    printf("Label file has two variables: fn, label \n");
    printf("useValidData: 0 --> not use, 1 --> use \n");
    printf("saveModelFile: random forest model file \n");
    printf("resultFile: validation result (confusion matrix) file \n");
}

static void readFramenumberFeatureLabel(const char * feature_mat_file,
                                        const char * label_mat_file,
                                        vector<int> & fns,
                                        vector<Eigen::VectorXf> & features,
                                        vector<int> & labels)
{
    assert(feature_mat_file);
    assert(label_mat_file);
    
    Eigen::MatrixXi fn1, fn2;
    Eigen::MatrixXf feature, label;
    
    // read feature
    bool is_read = false;
    is_read = matio::readMatrix<Eigen::MatrixXi>(feature_mat_file, "fn", fn1);
    assert(is_read);
    is_read = matio::readMatrix<Eigen::MatrixXf>(feature_mat_file, "feature", feature);
    assert(is_read);
    
    // read label
    is_read = matio::readMatrix<Eigen::MatrixXi>(label_mat_file, "fn", fn2);
    assert(is_read);
    is_read = matio::readMatrix<Eigen::MatrixXf>(label_mat_file, "label", label);
    
    assert(fn1.rows() == fn2.rows());
    long N = fn1.rows();
    for (int i = 0; i<N; i++) {
        assert(fn1(i, 0) == fn2(i, 0));
        fns.push_back(fn1(i, 0));
        
        int lab = label(i, 0);
        assert(lab >= 0);
        labels.push_back(lab);
        
        Eigen::VectorXf feat = feature.row(i).transpose();
        features.push_back(feat);
    }
    assert(fns.size() == features.size());
    assert(fns.size() == labels.size());
    printf("read %ld feature and label examples.\n", fns.size());
}

int main(int argc, const char * argv[])
{
    if (argc != 9) {
        printf("argc is %d, should be 9 .\n", argc);
        help();
        return -1;
    }
    
    const char *train_feature_file = argv[1];
    const char *train_label_file   = argv[2];
    const char *valid_feature_file = argv[3];
    const char *valid_label_file = argv[4];
    const int use_valid_data = strtod(argv[5], NULL);
    const char *param_file = argv[6];
    const char *model_file = argv[7];
    const char *result_file = argv[8];
    
    
    /*
    const char *train_feature_file = "/Users/jimmy/Desktop/DTC_10_fold/10_fold/trainX_01.mat";
    const char *train_label_file   = "/Users/jimmy/Desktop/DTC_10_fold/10_fold/trainY_01.mat";
    const char *valid_feature_file = "/Users/jimmy/Desktop/DTC_10_fold/10_fold/testX_01.mat";
    const char *valid_label_file = "/Users/jimmy/Desktop/DTC_10_fold/10_fold/testY_01.mat";
    const int use_valid_data = 1;
    const char *param_file = "/Users/jimmy/Desktop/DTC_10_fold/dtc_tree_param.txt";
    const char *model_file = "model.txt";
    const char *result_file = "result.mat";
     */
    
    assert(use_valid_data == 0 || use_valid_data == 1);
    vector<int>  train_fns;
    vector<Eigen::VectorXf> train_features;
    vector<int>  train_labels;
    readFramenumberFeatureLabel(train_feature_file, train_label_file,
                                train_fns, train_features, train_labels);
    
    
    vector<int>  valid_fns;
    vector<Eigen::VectorXf> valid_features;
    vector<int>  valid_labels;
    if (use_valid_data) {
        readFramenumberFeatureLabel(valid_feature_file, valid_label_file,
                                    valid_fns, valid_features, valid_labels);
        printf("validation data is used \n");
    }
    
    DTCTreeParameter tree_param;
    bool is_read = tree_param.readFromFile(param_file);
    assert(is_read);
    tree_param.feature_dimension_ = (int)train_features[0].size();
    cout<<tree_param<<endl;
    
    
    DTClassifierBuilder builder;
    builder.setTreeParameter(tree_param);
    
    // train and save model
    DTClassifer model;
    builder.buildModel(model, train_features, train_labels, valid_features, valid_labels, model_file);
    model.save(model_file);
    printf("save model to %s\n", model_file);
    
    // test model using different number of trees
    
    if (use_valid_data) {
        
        vector<int> predictions;
        for (int i = 0; i<valid_features.size(); i++) {
            int pred = 0;
            bool is_pred = model.predict(valid_features[i], pred);
            assert(is_pred);
            predictions.push_back(pred);
        }
        assert(valid_fns.size() == predictions.size());
        
        
        Eigen::MatrixXd confusion = DTUtil::confusionMatrix(predictions, valid_labels, tree_param.category_num_, true);
        cout<<"Validation confusion matrix is: \n"<<confusion<<endl;
        
        Eigen::VectorXd accuracy = DTUtil::accuracyFromConfusionMatrix(confusion);
        cout<<"Validation accuracy is:\n "<<accuracy.transpose()<<endl;
        
        Eigen::MatrixXd gd_pred = Eigen::MatrixXd((int)predictions.size(), 2);
        for (int i = 0; i<valid_labels.size(); i++) {
            gd_pred(i, 0) = valid_labels[i];
            gd_pred(i, 1) = predictions[i];
        }
       
        // save prediction result and confusion matrix
        vector<string> names;
        vector<Eigen::MatrixXd> data;
        names.push_back("gd_pred");
        names.push_back("prediction_confusion");
        data.push_back(gd_pred);
        data.push_back(confusion);
        matio::writeMultipleMatrix<Eigen::MatrixXd>(result_file, names, data);
    }
    
    cout<<tree_param<<endl;
    
    return 0;
}
#endif
