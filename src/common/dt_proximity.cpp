//
//  dt_proximity.cpp
//  SequentialRandomForest
//
//  Created by jimmy on 2017-07-14.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "dt_proximity.hpp"
#include <vector>

using std::unordered_map;
using std::vector;

void DTProximity::addExample(const int index1, const int index2)
{
 //   assert(index1 >= 0 && index1 < prox_.rows());
 //   assert(index2 >= 0 && index2 < prox_.cols());
    assert(index1 >= 0);
    assert(index2 >= 0);
    assert(index1 != index2);
    
    int key = 0;
    if (index1 < index2) {
        key = this->indexToKey(index1, index2);
    }
    else {
        key = this->indexToKey(index2, index1);
    }
    
    if (count_.find(key) != count_.end()) {
        count_[key] += 1.0f;
    }
    else {
        count_[key] = 1.0f;
    }
}

namespace  {
    class IndexValue {
        public:
        
        int index;
        double value;
        
        IndexValue(int i, double v)
        {
            index = i;
            value = v;
        }
        
        bool operator < (const IndexValue& other) const
        {
            return value > other.value;  // decending order
        }
    };
};

void DTProximity::computeProximityMatrix(const int knn)
{
    assert(knn > 0);
    prox_ = Eigen::MatrixXf::Zero(num_, knn);
    
    vector<vector<IndexValue> > neighbors(num_);
    for (auto it = count_.begin(); it != count_.end(); it++) {
        int key = it->first;
        double val = it->second;
        int index1 = 0;
        int index2 = 0;
        this->keyToIndex(key, index1, index2);
        assert(index1 < num_);
        IndexValue iv(index2, val);
        neighbors[index1].push_back(iv);
    }
    
    // sort by value
    for (int i = 0; i<neighbors.size(); i++) {
        std::sort(neighbors[i].begin(), neighbors[i].end());
        for (int j = 0; j<neighbors[i].size() && j < knn; j++) {
            prox_(i, j) = neighbors[i][j].value;
        }
    }    
}

int DTProximity::indexToKey(const int index1, const int index2) const
{
    assert(index1 <= index2);
    return index1 * num_ + index2;
}

void DTProximity::keyToIndex(const int key, int & index1, int & index2) const
{
    assert(key >= 0);
    index1 = key/num_;
    index2 = key%num_;
}
