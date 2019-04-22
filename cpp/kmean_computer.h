//
// Created by Paul Renauld on 2019-04-22.
//

#ifndef CPP_KMEAN_COMPUTER_H
#define CPP_KMEAN_COMPUTER_H

#include <vector>
#include "Point.h"

typedef struct {
    size_t n;
    Point points[];
} Dataset;
typedef std::vector<Point> ClusterPosition;

class kmean_computer {
  public:
    kmean_computer(size_t k, Dataset *dataset) : k(k), dataset(dataset) {}

    virtual ClusterPosition converge() = 0;

  protected:
    const size_t k;
    const Dataset *dataset;

};

#endif //CPP_KMEAN_COMPUTER_H

