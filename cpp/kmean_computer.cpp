
#include "kmean_computer.h"
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
kmean_computer::~kmean_computer() {
  delete[] clusters;
  delete[] cluster_for_point;
}

kmean_computer::kmean_computer(size_t k, size_t n, Dataset dataset) :
        k(k), n(n), dataset(dataset) {
  clusters = new Point[k];
  cluster_for_point = new unsigned short[n];
}

ClusterPosition kmean_computer::converge() {
  init_starting_clusters();
  
  time_type t0 = Clock::now();
  while (update_cluster_for_point()) {
    update_cluster_positions();
  }
  after_converge();

  time_type t1 = Clock::now();
  long time = difftime(t0, t1);
  std::cout << "Total time: " << time << "ms" << std::endl;
  return clusters; 
}

float kmean_computer::compute_silhouette() const {
  double avg = 0.0;
  for (size_t i = 0; i < n; ++i) {
    unsigned short cluster = cluster_for_point[i];
    Point &point = dataset[i];
    float mean_to_clust[k];
    size_t cluster_count[k];

    for (size_t a = 0; a < k; ++a) {
      mean_to_clust[a] = 0.f;
      cluster_count[a] = 0;
    }

    for (size_t j = 0; j < n; ++j) {
      unsigned short cluster_j = cluster_for_point[j];
      float distance = sqrt(point.distance_squared_to(dataset[j]));

      mean_to_clust[cluster_j] += distance;
      cluster_count[cluster_j]++;
    }

    cluster_count[cluster]--;

    float b_i = HUGE_VALF;
    float a_i = 0.0;

    for (size_t c = 0; c < k; ++c) {
      mean_to_clust[c] /= cluster_count[c];

      if (c == cluster) {
        a_i = mean_to_clust[c];
      } else if (mean_to_clust[c] < b_i) {
        b_i = mean_to_clust[c];
      }
    }

    float denom = std::max(a_i, b_i);
    float s = (b_i - a_i) / denom;
    avg += s;
  }

  avg /= n;
  return avg;
}

std::ostream &operator<<(std::ostream &os, const kmean_computer &computer) {
  for (size_t curr_k = 0; curr_k < computer.k; ++curr_k) {
    os << "C," << computer.clusters[curr_k] << std::endl;
    for (size_t i = 0; i < computer.n; ++i) {
      if (computer.cluster_for_point[i] >= computer.k) os << "out of bound cluster " <<  computer.cluster_for_point[i] << std::endl;
      if (computer.cluster_for_point[i] == curr_k) {
        os << computer.dataset[i] << std::endl;
      }
    }
  }
  return os;
}


