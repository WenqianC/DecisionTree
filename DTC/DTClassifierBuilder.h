//
//  DTClassifierBuilder.h
//  Classifer_RF
//
//  Created by jimmy on 2016-09-20.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef __Classifer_RF__DTClassifierBuilder__
#define __Classifer_RF__DTClassifierBuilder__

#include <stdio.h>
#include "DTClassifier.h"


class DTClassifierBuilder
{
private:
    DTCTreeParameter tree_param_;
    
public:
    void setTreeParameter(const DTCTreeParameter & param);
    
    bool buildModel(DTClassifer & model,
                    const vector<VectorXd> & features,
                    const vector<unsigned int> & labels,
                    const char * modle_file_name = NULL) const;
    
    
    
    
};

#endif /* defined(__Classifer_RF__DTClassifierBuilder__) */
