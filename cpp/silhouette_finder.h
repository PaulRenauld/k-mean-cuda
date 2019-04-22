//
// Created by Paul Renauld on 2019-04-22.
//

#ifndef CPP_SILHOUETTE_FINDER_H
#define CPP_SILHOUETTE_FINDER_H

#include <iostream>
#include "data_structure.h"
#include "kmean_computer.h"
#include "seq_computer.h"

typedef seq_computer Computer;

class silhouette_finder {
  public:
    silhouette_finder(size_t n, Dataset dataset) : dataset(dataset), n(n) {}

    virtual ~silhouette_finder();

    Computer *find_best_k(size_t min = 1, size_t max = -1,
                               std::ostream *out = nullptr);

    float try_k(size_t k);

  private:
    const Dataset dataset;
    const size_t n;
    Computer *best_cluster;
    float best_silhouette = -2;

};


#endif //CPP_SILHOUETTE_FINDER_H
