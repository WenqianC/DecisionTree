//
//  bt_rnd_regressor_builder.h
//  RGBD_RF
//
//  Created by jimmy on 2017-01-21.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __RGBD_RF__bt_rnd_regressor_builder__
#define __RGBD_RF__bt_rnd_regressor_builder__

#include <stdio.h>
#include "bt_rnd_regressor.h"
#include "bt_rnd_util.h"

class BTRNDRegressorBuilder
{
    typedef BTRNDRegressor Regressor;
    typedef SCRFRandomFeature Feature;
private:
    BTRNDTreeParameter tree_param_;
    DatasetParameter dataset_param_;
    
    typedef BTRNDTree TreeType;
    
public:
    void setTreeParameter(const BTRNDTreeParameter & param);
    void setDatasetParameter(const DatasetParameter & param);
    
    bool buildModel(Regressor & model,
                    const vector<Feature> & features,
                    const vector<VectorXf> & labels,
                    const vector<cv::Mat> & rgb_images,
                    const int max_check,
                    const char * model_file_name = NULL) const;
    
    // build model from subset of images without dropout
    // release_memory: true  when the tree is very large
    bool buildModel(Regressor& model,
                    const vector<string> & rgb_img_files,
                    const vector<string> & depth_img_files,
                    const vector<string> & pose_files,
                    const int max_check,
                    const bool release_memory = true,
                    const char *model_file_name = NULL) const;  
    
    
private:
    bool testValidataionError(const BTRNDTree & tree,
                              const vector<string> & rgb_img_files,
                              const vector<string> & depth_img_files,
                              const vector<string> & pose_files,
                              const int sample_frame_num,
                              const int max_check,
                              const double error_threshold) const;
    
};


#endif /* defined(__RGBD_RF__bt_rnd_regressor_builder__) */
