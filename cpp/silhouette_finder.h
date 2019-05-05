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
  #include "par_computer.h"
  #define COMPUTER_TYPE "Parallel"
  typedef par_computer Computer;
#else
  #include "seq_computer.h"
  #define COMPUTER_TYPE "Sequential"
  typedef seq_computer Computer;
#endif


class silhouette_finder {
  public:
    silhouette_finder(size_t n, Dataset dataset, 
                      bool approx_silhouette = false, bool use_omp = false)
            :
            dataset(dataset), n(n), 
            approx_silhouette(approx_silhouette), use_omp(use_omp),
            best_cluster(nullptr), best_silhouette(-2) {}

    virtual ~silhouette_finder();

    Computer *find_best_k(size_t min, size_t max, size_t step,
                               std::ostream *out = nullptr);

    float try_k(size_t k, std::ostream *out = nullptr);

  private:
    const Dataset dataset;
    const size_t n;
    const bool approx_silhouette, use_omp;
    Computer *best_cluster;
    float best_silhouette;

};


#endif //CPP_SILHOUETTE_FINDER_H
