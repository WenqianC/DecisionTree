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

#if 0
static void help()
{
    printf("program              modelFile   featureFileList        labelFileList           sample_ratio \n");
    printf("Seq_DTC_random_test  model.txt   feature_file_list.txt  label_file_list.txt     0.1          \n");
    printf("feature_file format: fn1 feat1 fn2 feat2 ... in a single file \n");
    printf("label_file: fn1 label1 fn2 label2 ... in a single file \n");
    printf("sample_ratio: 0.1, random sample ratio \n");
    printf("output: a confusion matrix \n");
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
    const double sample_ratio = strtod(argv[4], NULL);
    
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
   
    model.predictDebug(fns1, features, labels, sample_ratio);
    
    
    return 0;
}
#endif
