//
// Created by Paul Renauld on 2019-04-22.
//

#include "silhouette_finder.h"


Computer *
silhouette_finder::find_best_k(size_t min, size_t max, std::ostream *out) {
  if (max == -1) max = n;

  for (size_t k = min; k < max; ++k) {
    float sil = try_k(k);
    if (out != nullptr) {
      *out << "silhouette with k=" << k << ": " << sil << std::endl;
    }
  }

  return best_cluster;
}

float silhouette_finder::try_k(size_t k) {
  Computer *computer = new Computer(k, n, dataset);
  computer->converge();
  float sil = computer->compute_silhouette();
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
