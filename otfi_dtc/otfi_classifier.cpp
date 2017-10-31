//
//  otfi_classifier.cpp
//  SequentialRandomForest
//
//  Created by jimmy on 2017-07-17.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "otfi_classifier.hpp"
#include <stack>
#include "dt_util.hpp"

using std::stack;

OTFIClassifier::OTFIClassifier()
{

}
OTFIClassifier::~OTFIClassifier()
{

}

bool OTFIClassifier::predict(const Eigen::VectorXf & feature,
                             int & prediction) const
{
    assert(trees_.size() > 0);
    assert(feature_dim_ == feature.size());

    VectorXf prob = Eigen::VectorXf::Zero(tree_param_.category_num_);
    for (int i = 0; i<trees_.size(); i++) {
        Eigen::VectorXf p;
        trees_[i]->predict(feature, p);
        assert(p.size() == prob.size());
        prob += p;
    }
    prob /= trees_.size();
    prob.maxCoeff(&prediction);
    return true;
}

static bool isSameValue(const float v1, const float v2)
{
    return fabsf(v1 - v2) < 0.00001;
}

bool OTFIClassifier::imputeFeature(const vector<Eigen::VectorXf> & features,
                       const vector<int> & labels,                       
                       vector<Eigen::VectorXf> & mdata_features,  // output
                       const vector<int> & mdata_labels,                       
                       const float mdata_mask) const
{
    assert(features.size() == labels.size());
    assert(mdata_features.size() == mdata_labels.size());
    assert(trees_.size() > 0);

    const vector<int> indices = DTUtil::range<int>(0, (int)features.size(), 1);
    const vector<int> mdata_indices = DTUtil::range<int>(0, (int)mdata_features.size(), 1);
    const int tree_num = (int)trees_.size();
    const int mdata_num = (int)mdata_features.size();
    vector<vector<Eigen::VectorXf> > imputed_features(tree_num);
    vector<vector<float> > imputation_weight(tree_num);
    // imputate feautre using each tree
    for(int i = 0; i<tree_num; i++) {       
        imputed_features[i] = mdata_features;
        imputation_weight[i] = vector<float>(mdata_num, 0);
        trees_[i]->imputeFeature(features, labels, indices, 
                                 mdata_features, mdata_labels, mdata_indices, mdata_mask,
                                 imputed_features[i], imputation_weight[i]);        
    }

    // average imputation
    int zero_wt_num = 0;
    const int dims = (int)mdata_features[0].size();
    for(int i = 0; i<mdata_num; i++) {        
        // feature dimension
        for(int d = 0; d<mdata_features[i].size(); d++) {
            if (isSameValue(mdata_features[i][d], mdata_mask)) {
                float wt = 0.0f;
                float value = 0.0f;
                for (int k = 0; k<tree_num; k++) {
                    wt += imputation_weight[k][i];
                    value += imputed_features[k][i][d] * imputation_weight[k][i];
                }
                //printf("imputation weight is %lf\n", wt);
                if (wt == 0.0f) {
                    // randomly choose one tree imputation result
                    //printf("warning: imputatin weight is 0\n");
                    zero_wt_num++;
                    mdata_features[i][d] = imputed_features[rand()%tree_num][i][d];
                }
                else {
                    value /= wt;
                    mdata_features[i][d] = value;
                }
            } 
        }
    }
    printf("zero weight percentage is %lf\n", 1.0*zero_wt_num/(mdata_num*dims));

    return true;
}

bool OTFIClassifier::save(const char *fileName) const
{
    assert(trees_.size() > 0);
    // write tree number and tree files to file Name
    FILE *pf = fopen(fileName, "w");
    if(!pf) {
        printf("Error: can not open file %s\n", fileName);
        return false;
    }
    
    tree_param_.writeToFile(pf); // must put at the very begining
    fprintf(pf, "%d %d\n", feature_dim_, tree_param_.category_num_);
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
            OTFITreeNode::writeTree(tree_files[i].c_str(), trees_[i]->root_);
        }
    }
    fclose(pf);
    printf("save to %s\n", fileName);
    return true;
}

bool OTFIClassifier::load(const char *fileName)
{
    FILE *pf = fopen(fileName, "r");
    if (!pf) {
        printf("Error: can not open file %s\n", fileName);
        return false;
    }
    
    bool is_read = tree_param_.readFromFile(pf);
    assert(is_read);
    tree_param_.printSelf();
    
    int category_num = 0;
    int ret_num = fscanf(pf, "%d %d", &feature_dim_, &category_num);
    assert(ret_num == 2);
    
    vector<string> tree_files;
    for (int i = 0; i<tree_param_.tree_num_; i++) {
        char buf[1024] = {NULL};
        fscanf(pf, "%s", buf);
        tree_files.push_back(string(buf));
    }
    fclose(pf);
    
    for (int i = 0; i<trees_.size(); i++) {
        if (trees_[i]) {
             delete trees_[i];
            trees_[i] = 0;
        }      
    }
    trees_.clear();
    
    // read each tree
    for (int i = 0; i<tree_files.size(); i++) {
        Node * root = NULL;
        bool is_read = false;       
        is_read = Node::readTree(tree_files[i].c_str(), category_num, root);
        assert(is_read);
        assert(root);
        
        TreePtr tree = new Tree();
        tree->root_ = root;
        tree->setTreeParameter(tree_param_);       
        trees_.push_back(tree);
    }
    printf("read from %s\n", fileName);
    
    return true;
}

