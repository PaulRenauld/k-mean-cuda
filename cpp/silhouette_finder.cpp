//
// Created by Paul Renauld on 2019-04-22.
//

#include "silhouette_finder.h"

#include <thread>
#include <chrono>
#include <iostream>
#include <stdio.h>


typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::milliseconds milliseconds;
typedef Clock::time_point time_type;

inline long difftime(time_type t0, time_type t1) {
  milliseconds ms = std::chrono::duration_cast<milliseconds>(t1 - t0);
  return ms.count();
}


Computer *
silhouette_finder::find_best_k(size_t min, size_t max, size_t step, std::ostream *out) {
  if (max == -1) max = n;

  time_type t0 = Clock::now();

  for (size_t k = min; k < max; k += step) {
    float sil = try_k(k);
    if (out != nullptr) {
      *out << "silhouette with k=" << k << ": " << sil << std::endl;
    }
  }

  time_type t1 = Clock::now();
  long time = difftime(t0, t1);
  if (out != nullptr) {
    *out << "Total time to find best k: " << time << "ms" << std::endl;
  }

  return best_cluster;
}

float silhouette_finder::try_k(size_t k) {
  Computer *computer = new Computer(k, n, dataset);
  computer->converge();
  float sil = approx_silhouette ? computer->compute_silhouette_approximation()
                                : computer->compute_silhouette_all();
  if (sil > best_silhouette) {
    best_silhouette = sil;
    delete best_cluster;
    best_cluster = computer;
  } else {
    delete computer;
  }
  return sil;
}

silhouette_finder::~silhouette_finder() {
  delete best_cluster;
}
