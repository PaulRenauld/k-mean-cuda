//
// Created by Paul Renauld on 2019-04-22.
//

#ifndef CPP_SILHOUETTE_FINDER_H
#define CPP_SILHOUETTE_FINDER_H

#include <iostream>
#include <string>
#include "data_structure.h"
#include "kmean_computer.h"

#if CUDA==1
  #warning "Using cuda"
  #include "par_computer.h"
  #define COMPUTER_TYPE "Parallel"
  typedef par_computer Computer;
#else
  #warning "Using seq"
  #include "seq_computer.h"
  #define COMPUTER_TYPE "Sequential"
  typedef seq_computer Computer;
#endif


class silhouette_finder {
  public:
    silhouette_finder(size_t n, Dataset dataset, bool approx_silhouette = false)
            :
            dataset(dataset), n(n), approx_silhouette(approx_silhouette),
            best_cluster(nullptr), best_silhouette(-2) {}

    virtual ~silhouette_finder();

    Computer *find_best_k(size_t min, size_t max, size_t step,
                               std::ostream *out = nullptr);

    float try_k(size_t k);

  private:
    const Dataset dataset;
    const size_t n;
    const bool approx_silhouette;
    Computer *best_cluster;
    float best_silhouette;

};


#endif //CPP_SILHOUETTE_FINDER_H
