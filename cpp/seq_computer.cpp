//
// Created by Paul Renauld on 2019-04-22.
//

#include "seq_computer.h"
#include <iostream>
#include <set>
#include <iterator>
#include <random>


void seq_computer::init_starting_clusters() {
  std::default_random_engine generator;
  std::uniform_int_distribution<size_t> distribution(0, n - 1);

  std::set<size_t, std::greater<size_t>> positions;
  while (positions.size() < k) {
    positions.insert(distribution(generator));
  }

  size_t i = 0;
  for (auto const& index: positions) {
    clusters[i++] = dataset[index];
  }
}

void seq_computer::update_cluster_positions() {
  size_t count[k];

  Point p0 = Point();
  for (size_t j = 0; j < k; ++j) {
    clusters[j] = p0;
    count[j] = 0;
  }

  for (size_t i = 0; i < n; ++i) {
    int cluster= cluster_for_point[i];
    count[cluster]++;
    clusters[cluster] += dataset[i];
  }

  for (size_t j = 0; j < k; ++j) {
    clusters[j] /= count[j];
  }
}

bool seq_computer::update_cluster_for_point() {
  bool change = false;

  for (size_t i = 0; i < n; i++) {
    Point& datapoint = dataset[i];
    float min = datapoint.distance_squared_to(clusters[0]);
    unsigned short index_min = 0;

    for (unsigned short j = 1; j < k; j++) {
      float distance = datapoint.distance_squared_to(clusters[j]);
      if (distance < min) {
        min = distance;
        index_min = j;
      }
    }

    if (cluster_for_point[i] != index_min) {
      cluster_for_point[i] = index_min;
      change = true;
    }
  }

  return change;
}

float seq_computer::compute_silhouette() const {
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
      if (cluster_count[c] > 0) {
        mean_to_clust[c] /= cluster_count[c];
      } else {
        mean_to_clust[c] = 0;
      }

      if (c == cluster) {
        a_i = mean_to_clust[c];
      } else if (mean_to_clust[c] < b_i && mean_to_clust[c] != 0) {
        b_i = mean_to_clust[c];
      }
    }

    float denom = std::max(a_i, b_i);
    float s = (b_i - a_i) / denom;
    avg += s;
    if (isnan(s)) {
      std::cout << "NAN avg" << s << " " << a_i << " " << b_i << std::endl;
    }
  }

  avg /= n;
  return avg;
}

void seq_computer::after_converge() {}

seq_computer::~seq_computer() = default;
