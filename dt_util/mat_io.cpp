//
//  mat_io.cpp
//  SequentialRandomForest
//
//  Created by jimmy on 2017-07-01.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "mat_io.hpp"
#ifdef __cplusplus
extern "C" {
    #include "matio.h"
#endif

#ifdef __cplusplus
}
#endif  // closing brace for extern "C"

using Eigen::Matrix;

namespace matio {
    
    template<class matrixT>
    bool readMatrix(const char *file_name, const char *var_name, matrixT & mat_data)
    {
        assert(file_name);
        assert(var_name);
        
        mat_t    *matfp = NULL;
        matvar_t *matvar = NULL;
        bool is_read = false;
        
        matfp = Mat_Open(file_name, MAT_ACC_RDONLY);
        if ( NULL == matfp ) {
            printf("Error: opening MAT file \"%s\"!\n", file_name);
            return false;
        }
        matvar = Mat_VarRead(matfp, var_name);
        if ( NULL == matvar ) {
            printf("Error: Variable %s not found, or error reading MAT file",
                    var_name);
        }
        if (matvar->rank != 2) {
            printf("Error: Variable %s is not a matrix!\n", var_name);
        }
        else {
            size_t rows = matvar->dims[0];
            size_t cols = matvar->dims[1];
            void *data = matvar->data;
            assert(data);
            matio_types data_type = matvar->data_type;
            
            switch (data_type) {
                case MAT_T_DOUBLE:
                {
                    mat_data = matrixT::Zero(rows, cols);
                    // colum wise
                    double *pdata = (double *)data;
                    // copy data
                    for (int c = 0; c<cols; c++ ) {
                        double * p = &pdata[c * rows];
                        for (int r = 0; r<rows; r++) {
                            mat_data(r, c) = p[r];
                        }
                    }
                    is_read = true;
                }                    
                break;
                    
                case MAT_T_SINGLE:
                {
                    mat_data = matrixT::Zero(rows, cols);
                    float *pdata = (float *)data;
                    // copy data
                    for (int c = 0; c<cols; c++ ) {
                        float * p = &pdata[c * rows];
                        for (int r = 0; r<rows; r++) {
                            mat_data(r, c) = p[r];
                        }
                    }

                    is_read = true;
                }
                break;
                    
                default:
                    printf("Error: non-supported data type\n");
                    break;
            }
        }
        
        // free data
        if (matvar != NULL) {
            Mat_VarFree(matvar);
            matvar = NULL;
        }
        if (matfp != NULL) {
            Mat_Close(matfp);
        }
        if (is_read) {
            printf("read a %ld x %ld matrix named %s. \n", mat_data.rows(), mat_data.cols(), var_name);
        }
        return is_read;
    }
    
    template
    bool readMatrix(const char *file_name, const char *var_name, Eigen::MatrixXd& mat_data);
   
    template
    bool readMatrix(const char *file_name, const char *var_name, Eigen::MatrixXf& mat_data);
    
    template
    bool readMatrix(const char *file_name, const char *var_name, Eigen::MatrixXi& mat_data);
    
    
    
    
} // namespace matio
