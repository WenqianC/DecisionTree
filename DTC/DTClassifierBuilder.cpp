//
//  DTClassifierBuilder.cpp
//  Classifer_RF
//
//  Created by jimmy on 2016-09-20.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#include "DTClassifierBuilder.h"
#include "DTRandom.h"
#include <iostream>

using std::cout;
using std::endl;

void DTClassifierBuilder::setTreeParameter(const DTCTreeParameter & param)
{
    tree_param_ = param;
}

bool DTClassifierBuilder::buildModel(DTClassifer & model,
                                     const vector<VectorXd> & features,
                                     const vector<unsigned int> & labels,
                                     const char * modle_file_name) const
{
    assert(features.size() == labels.size());
    
    model.tree_param_ = tree_param_;
    model.trees_.clear();
    
    const int tree_num = tree_param_.tree_num_;
    for (int n = 0; n<tree_num; n++) {
        // bagging
        vector<unsigned int> training_indices;
        vector<unsigned int> validation_indices;
        DTRandom::outof_bag_sampling((unsigned int) features.size(), training_indices, validation_indices);
        
        
        DTCTree * tree = new DTCTree();
        assert(tree);
        double tt = clock();
        tree->buildTree(features, labels, training_indices, tree_param_);
        model.trees_.push_back(tree);
        
        printf("build tree %d cost %lf minutes\n", n, (clock()- tt)/CLOCKS_PER_SEC/60.0);
        
        // test on the validation data
        
        vector<Eigen::VectorXd> cv_probs;
        vector<unsigned int> cv_labels;
        for (int i = 0; i<validation_indices.size(); i++) {
            const int index = validation_indices[i];
            Eigen::VectorXd prob;
            model.predict(features[index], prob);
            cv_probs.push_back(prob);
            cv_labels.push_back(labels[index]);
        }
        
        Eigen::MatrixXd cv_conf = DTCUtil::confusionMatrix(cv_probs, cv_labels);       
        
        cout<<"out of bag cross validation confusion matrix: \n"<<cv_conf<<endl;
    }
    
    
    
    
    
    
    
    
    return true;
}