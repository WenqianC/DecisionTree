//
//  DTCTree.h
//  Classifer_RF
//
//  Created by jimmy on 2016-07-26.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef __Classifer_RF__DTCTree__
#define __Classifer_RF__DTCTree__

#include <stdio.h>
#include <vector>
#include <Eigen/Dense>
#include "DTCUtil.h"

using std::vector;
using Eigen::VectorXd;

class DTCNode;
// decision tree classifier Tree
class DTCTree
{
    friend class DTClassifer;
    
    DTCNode * root_;
    DTCTreeParameter tree_param_;
    
public:
    DTCTree(){root_ = NULL;}
    ~DTCTree(){;}
    
    // features:
    // labels: 0 - N-1
    // indices:
    bool buildTree(const vector<VectorXd> & features,
                   const vector<unsigned int> & labels,
                   const vector<unsigned int> & indices,
                   const DTCTreeParameter & param);
    
    bool predict(const Eigen::VectorXd & feature,
                 Eigen::VectorXd & prob) const;
    
    bool predict(const Eigen::VectorXd & feature,
                 unsigned int & pred) const;
    
    const DTCTreeParameter & getTreeParameter(void) const;
    void setTreeParameter(const DTCTreeParameter & param);
    
    
private:
    bool configureNode(const vector<VectorXd> & features,
                       const vector<unsigned int> & labels,
                       const vector<unsigned int> & indices,
                       DTCNode * node);
    
    bool predict(const DTCNode * node,
                 const Eigen::VectorXd & feature,
                 Eigen::VectorXd & prob) const;   
    
    
};


#endif /* defined(__Classifer_RF__DTCTree__) */
