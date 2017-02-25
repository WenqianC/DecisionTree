//
//  seq_dtc_train_main.cpp
//  Classifer_RF
//
//  Created by jimmy on 2017-02-22.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include <stdio.h>
#include "DTUtil_IO.h"
#include "seq_dt_classifier.h"
#include "seq_dt_classifier_builder.h"
#include "dt_util.h"
#include <iostream>

using namespace::std;

#if 0
static void help()
{
    printf("program       modelFile   featureFileList        labelFileList           save_file    \n");
    printf("Seq_DTC_test  model.txt   feature_file_list.txt  label_file_list.txt     fn_gd_pred   \n");
    printf("feature_file format: fn1 feat1 fn2 feat2 ... in a single file \n");
    printf("label_file: fn1 label1 fn2 label2 ... in a single file \n");
    printf("test frame number can be discontinuous. \n");
    printf("output: a confusion matrix and average accuracy. \n");
}

int main(int argc, const char * argv[])
{
    if (argc != 5) {
        printf("argc is %d, should be 5.\n", argc);
        help();
        return -1;
    }
    
    const char *model_file = argv[1];
    const char *feature_file_list = argv[2];
    const char *label_file_list   = argv[3];
    const char *save_file = argv[4];
    /*
    const char *model_file = "/Users/jimmy/Desktop/Sec_DTC_app/model/debug.txt";
    const char *feature_file_list = "/Users/jimmy/Desktop/Sec_DTC_app/test_files/feature_file.txt";
    const char *label_file_list   = "/Users/jimmy/Desktop/Sec_DTC_app/test_files/label_file.txt";
    const char *save_file = "fn_gd_pred.txt";
     */
    
    // read model
    SeqDTClassifier model;
    bool is_read = model.load(model_file);
    assert(is_read);
    
    // read feature files
    vector<string> feature_files, label_files;
    DTUtil_IO::read_files(feature_file_list, feature_files);
    DTUtil_IO::read_files(label_file_list, label_files);
    assert(feature_files.size() == 1);
    assert(label_files.size() == 1);
    
    printf("feature label files\n");
    for (int i =0; i<feature_files.size(); i++) {
        printf("%s \n", feature_files[i].c_str());
        printf("%s \n", label_files[i].c_str());
    }
    
    vector<int> fns1;
    vector<int> fns2;
    vector<Eigen::VectorXf> features;
    vector<unsigned int> labels;
    
    is_read = DTUtil_IO::read_fn_matrix(feature_files[0].c_str(), fns1, features);
    assert(is_read);
    is_read = DTUtil_IO::read_fn_labels(label_files[0].c_str(), fns2, labels);
    assert(is_read);
    assert(fns1.size() == fns2.size());
    
    // check frame number
    for (int i = 0; i<fns1.size(); i++) {
        assert(fns1[i] == fns2[i]);
    }
    
    assert(features.size() == labels.size());
    assert(features.size() > 0);
    
    vector<unsigned int> predictions;
    model.multipleSequencePredict(fns1, features, predictions);
    assert(fns1.size() == predictions.size());
    
    Eigen::MatrixXi result((int)predictions.size(), 3);
    for (int i = 0; i<predictions.size(); i++) {
        result(i, 0) = fns1[i];
        result(i, 1) = (int)labels[i];
        result(i, 2) = (int)predictions[i];
    }
    
    Eigen::MatrixXd confusion = DTUtil::confusionMatrix(predictions, labels, model.categoryNum(), false);
    cout<<"Seq DTC prediction confusion matrix is: \n"<<confusion<<endl;
    
    Eigen::VectorXd accuracy = DTUtil::accuracyFromConfusionMatrix(confusion);
    cout<<"accuracy is:\n "<<accuracy.transpose()<<endl;
    
    DTUtil_IO::save_matrix<MatrixXi>(save_file, result);
    
    return 0;
}
#endif
