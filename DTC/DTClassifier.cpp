//
//  DTClassifier.cpp
//  Classifer_RF
//
//  Created by jimmy on 2016-09-20.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#include "DTClassifier.h"
#include "DTCNode.h"


bool DTClassifer::predict(const Eigen::VectorXd & feature,
                          Eigen::VectorXd & prob) const
{
    assert(trees_.size() > 0);
    
    const DTCTreeParameter param = trees_[0]->getTreeParameter();
    prob = Eigen::VectorXd::Zero(param.category_num_);
    
    for (int i = 0; i<trees_.size(); i++) {
        Eigen::VectorXd p;
        trees_[i]->predict(feature, p);
        prob += p;
    }
    prob /= trees_.size();
    
    return true;
}

bool DTClassifer::predict(const Eigen::VectorXd & feature,
                          unsigned int & pred)
{
    Eigen::VectorXd prob;
    bool isPred = this->predict(feature, prob);
    if (!isPred) {
        return false;
    }
    prob.maxCoeff(&pred);
    return true;
}

bool DTClassifer::save(const char *fileName) const
{
    assert(trees_.size() > 0);
    // write tree number and tree files to file Name
    FILE *pf = fopen(fileName, "w");
    if(!pf) {
        printf("Error: can not open file %s\n", fileName);
        return false;
    }
  //  fprintf(pf, "%d %d\n", feature_dim_, label_dim_);
    
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
            DTCNode::writeTree(tree_files[i].c_str(), trees_[i]->root_);
        }
    }
    fclose(pf);
    printf("save to %s\n", fileName);
    return true;
    
    return true;
}

bool DTClassifer::load(const char *fileName)
{
    FILE *pf = fopen(fileName, "r");
    if (!pf) {
        printf("Error: can not open file %s\n", fileName);
        return false;
    }
    
  //  int ret_num = fscanf(pf, "%d %d", &feature_dim_, &label_dim_);
  //  assert(ret_num);
    
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
        
        DTCNode * root = NULL;
        bool isRead = false;
        isRead = DTCNode::readTree(treeFiles[i].c_str(), root);
        assert(isRead);
        assert(root);
        
        DTCTree *tree = new DTCTree();
        tree->root_ = root;
        tree->setTreeParameter(tree_param_);
        trees_.push_back(tree);
        
    }
    printf("read from %s\n", fileName);
    return true;
     

    return false;
    
}