//
//  otfi_classifier_builder.cpp
//  SequentialRandomForest
//
//  Created by jimmy on 2017-07-17.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "otfi_classifier_builder.hpp"
#include "dt_random.hpp"
#include <iostream>
#include "dt_util.hpp"

using std::cout;
using std::endl;

OTFIClassifierBuilder::OTFIClassifierBuilder()
{

}

OTFIClassifierBuilder::~OTFIClassifierBuilder()
{

}

void OTFIClassifierBuilder::setTreeParameter(const TreeParameter & param)
{
    tree_param_ = param;
}

void OTFIClassifierBuilder::buildModel(OTFIClassifier & model, 
                                       const vector<Eigen::VectorXf> & features,
                                       const vector<int> & labels,
                                       const char* model_file_name)
{
    assert(features.size() == labels.size());    
    model.tree_param_ = tree_param_;
    model.trees_.clear();   // @todo release memory
    model.feature_dim_ = (int)features[0].size();    
    
    const int tree_num = tree_param_.tree_num_;
    const int category_num = tree_param_.category_num_;
    for (int n = 0; n<tree_num; n++) {
        // Step 1. out of bag sampling
        vector<int> training_indices;
        vector<int> oob_indices;
        DTRandom::outofBagSampling<int>((unsigned int) features.size(), training_indices, oob_indices);      
                
        // training
        TreePtr pTree = new TreeType();
        assert(pTree);
        double tt = clock();
        pTree->buildTree(features, labels, training_indices, tree_param_);
        model.trees_.push_back(pTree);
        printf("build tree %d cost %lf minutes\n", n, (clock()- tt)/CLOCKS_PER_SEC/60.0);
        
        
        if (model_file_name != NULL) {
            model.save(model_file_name);
        }
        
        // single tree out-of-bag validataion error
        vector<int> oob_predictions;
        vector<int> oob_labels;
        for (int i = 0; i<oob_indices.size(); i++) {
            int index = oob_indices[i];
            int pred = 0;
            bool is_pred = pTree->predict(features[index], pred);
            assert(is_pred);
            if (is_pred) {
                oob_predictions.push_back(pred);
                oob_labels.push_back(labels[index]);
            }
        }        
        Eigen::MatrixXd confusion = DTUtil::confusionMatrix(oob_predictions, oob_labels, category_num, true);
        cout<<"cross validation confusion matrix: \n"<<confusion<<endl;
    }

    printf("build model done %lu trees.\n", model.trees_.size());    
}