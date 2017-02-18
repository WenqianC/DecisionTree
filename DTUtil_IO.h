//
//  DTUtil_IO.h
//  Classifer_RF
//
//  Created by jimmy on 2016-10-04.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef __Classifer_RF__DTUtil_IO__
#define __Classifer_RF__DTUtil_IO__

#include <stdio.h>
#include <vector>
#include <Eigen/Dense>
#include <string>
#include <fstream>

using std::vector;
using std::string;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::MatrixXi;

class DTUtil_IO
{
public:
    static bool read_matrix(const char * file_name, vector<VectorXd> & data);
    static bool read_matrix(const char * file_name, Eigen::MatrixXd & data);
    static bool save_matrix(const char * file_name, const vector<VectorXd> & data);
    static bool read_labels(const char * file_name, vector<unsigned int> & labels);
    
    // files with frame number as the first column
    static bool read_fn_matrix(const char *file_name, vector<int> & fns, vector<Eigen::VectorXd> & data);
    static bool read_fn_labels(const char * file_name, vector<int> & fns, vector<unsigned int> & labels);
    static bool read_fn_gd_preds(const char *file_name, vector<int> & fns, vector<unsigned int> & gds,  vector<unsigned int> & preds);
    
    static bool read_files(const char *file_name, vector<string> & files);
    static bool write_files(const char *file_name, const vector<string>& files);
    
    //
    template<class T>
    static bool save_matrix(const char * file_name, const T &m)
    {
        std::ofstream file(file_name);
        if (file.is_open()) {
            int rows = (int)m.rows();
            int cols = (int)m.cols();
            file<<rows<<" "<<cols<<"\n"<<m<<"\n";
            printf("save to %s\n", file_name);
            return true;
        }
        else {
            printf("Error: can not open file %s\n", file_name);
            return false;
        }
    }    
};




#endif /* defined(__Classifer_RF__DTUtil_IO__) */
