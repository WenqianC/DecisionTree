//
//  DTClassifier.cpp
//  Classifer_RF
//
//  Created by jimmy on 2016-09-20.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#include "dt_classifier.h"
#include "dt_util.hpp"


bool DTClassifier::predict(const Eigen::VectorXf & feature,
                          Eigen::VectorXf & prob) const
{
    assert(trees_.size() > 0);
    
    const DTCTreeParameter param = trees_[0]->getTreeParameter();
    prob = Eigen::VectorXf::Zero(param.category_num_);
    
    for (int i = 0; i<trees_.size(); i++) {
        Eigen::VectorXf p;
        trees_[i]->predict(feature, p);
        prob += p;
    }
    prob /= trees_.size();
    
    return true;
}

bool DTClassifier::predict(const Eigen::VectorXf & feature,
                          int & pred)
{
    Eigen::VectorXf prob;
    bool isPred = this->predict(feature, prob);
    if (!isPred) {
        return false;
    }
    prob.maxCoeff(&pred);
    return true;
}

Eigen::VectorXd DTClassifier::measureVariableImportance(const vector<Eigen::VectorXf> & features,
                                                        const vector<int> & labels)
{
    assert(features.size() == labels.size());
    assert(trees_.size() > 0);
    
    const int dims = (int)features[0].size();
    const int category_num = tree_param_.category_num_;
    const int n = (int)features.size();
    Eigen::VectorXd vp = Eigen::VectorXd::Zero(dims, 1);
    
    // step 1: prediction using original data
    vector<int> predictions;
    for (int i = 0; i<features.size(); i++) {
        int pred = 0;
        this->predict(features[i], pred);
        predictions.push_back(pred);
    }
    assert(predictions.size() == labels.size());
    Eigen::MatrixXd confusion = DTUtil::confusionMatrix<int>(predictions, labels, category_num, false);
    Eigen::VectorXd accuracy  = DTUtil::accuracyFromConfusionMatrix(confusion);
    assert(accuracy.size() == category_num + 1);
    
    double org_acc = accuracy[category_num];
    vector<double> one_dim_values(n);
    DTRandom rnd_generator;
    for (int d = 0; d < dims; d++) {
        vector<Eigen::VectorXf> modified_features = features;
        for (int i = 0; i<n; i++) {
            one_dim_values[i] = features[i][d];
        }
        double min_v = *std::min_element(one_dim_values.begin(), one_dim_values.end());
        double max_v = *std::max_element(one_dim_values.begin(), one_dim_values.end());
        vector<double> rand_values =rnd_generator.getRandomNumbers(min_v, max_v, n);
        
        // permuate feature by randomly generated values
        for (int i = 0; i<n; i++) {
            modified_features[i][d] = (float)rand_values[i];
        }
        
        // prediction using new feature
        vector<int> predictions;
        for (int i = 0; i<n; i++) {
            int pred = 0;
            this->predict(modified_features[i], pred);
            predictions.push_back(pred);
        }
        assert(predictions.size() == labels.size());
        Eigen::MatrixXd confusion = DTUtil::confusionMatrix<int>(predictions, labels, category_num, false);
        Eigen::VectorXd accuracy  = DTUtil::accuracyFromConfusionMatrix(confusion);
        assert(accuracy.size() == category_num + 1);
        
        // measure accuracy (not always) decresement
        double cur_acc = accuracy[category_num];
        vp[d] = org_acc - cur_acc;
    }
    
    return vp;
}

bool DTClassifier::save(const char *fileName) const
{
    assert(trees_.size() > 0);
    // write tree number and tree files to file Name
    FILE *pf = fopen(fileName, "w");
    if(!pf) {
        printf("Error: can not open file %s\n", fileName);
        return false;
    }
    
    tree_param_.writeToFile(pf);
    vector<string> tree_files;
    string baseName = string(fileName);
    baseName = baseName.substr(0, baseName.size()-4);
    for (int i = 0; i<trees_.size(); i++) {
        char buf[1024] = {NULL};
        sprintf(buf, "_%08d", i);
        string fileName = baseName + string(buf) + string(".txt");
        fprintf(pf, "%s\n", fileName.c_str());
        tree_files.push_back(fileName);
    }
    
    for (int i = 0; i<trees_.size(); i++) {
        if (trees_[i]) {
            trees_[i]->writeTree(tree_files[i].c_str());
        }
    }
    fclose(pf);
    printf("save to %s\n", fileName);
    return true;
}

bool DTClassifier::load(const char *fileName)
{
    FILE *pf = fopen(fileName, "r");
    if (!pf) {
        printf("Error: can not open file %s\n", fileName);
        return false;
    }
    
    bool is_read = tree_param_.readFromFile(pf);
    assert(is_read);
   // tree_param_.printSelf();   
    
    vector<string> treeFiles;
    for (int i = 0; i<tree_param_.tree_num_; i++) {
        char buf[1024] = {NULL};
        fscanf(pf, "%s", buf);
        treeFiles.push_back(string(buf));
    }
    fclose(pf);
    
    for (int i = 0; i<trees_.size(); i++) {
        delete trees_[i];
        trees_[i] = 0;
    }
    trees_.clear();
    
    // read each tree
    for (int i = 0; i<treeFiles.size(); i++) {        
        DTCTree *tree = new DTCTree();
        assert(tree);
        bool is_read = tree->readTree(treeFiles[i].c_str());
        assert(is_read);
        tree->setTreeParameter(tree_param_);
        trees_.push_back(tree);
        
    }
    printf("read from %s\n", fileName);
    return true;
}