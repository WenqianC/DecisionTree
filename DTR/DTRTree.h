//
//  DTRTree.h
//  Classifer_RF
//
//  Created by jimmy on 2016-10-03.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef __Classifer_RF__DTRTree__
#define __Classifer_RF__DTRTree__

#include <stdio.h>
#include <vector>
#include <Eigen/Dense>
#include "DTRUtil.h"

using std::vector;
using Eigen::VectorXd;

class DTRNode;
// decision tree regression Tree
class DTRTree
{
    friend class DTRegressor;
    
    DTRNode * root_;
    DTRTreeParameter tree_param_;
    
public:
    DTRTree(){root_ = NULL;}
    ~DTRTree(){;}
    
    // features:
    // labels: regression label
    // indices:
    bool buildTree(const vector<VectorXd> & features,
                   const vector<VectorXd> & labels,
                   const vector<unsigned int> & indices,
                   const DTRTreeParameter & param);
    
    
    bool predict(const Eigen::VectorXd & feature,
                 Eigen::VectorXd & pred) const;
    
    const DTRTreeParameter & getTreeParameter(void) const;
    void setTreeParameter(const DTRTreeParameter & param);
    
    int leafNodeNumber(void) const;
    
private:
    bool configureNode(const vector<VectorXd> & features,
                       const vector<VectorXd> & labels,
                       const vector<unsigned int> & indices,
                       DTRNode * node);
    
    
    bool predict(const DTRNode * node,
                 const Eigen::VectorXd & feature,
                 Eigen::VectorXd & pred) const;
    
    void leafNodeNumber(const DTRNode * node, int & num) const;
    
    
};


#endif /* defined(__Classifer_RF__DTRTree__) */
