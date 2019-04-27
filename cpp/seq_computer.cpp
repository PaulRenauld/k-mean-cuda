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
