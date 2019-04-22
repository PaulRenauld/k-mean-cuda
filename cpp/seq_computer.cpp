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
  auto random_index = std::bind ( distribution, generator );

  std::set<size_t, std::greater<>> positions;
  while (positions.size() < k) {
    positions.insert(random_index());
  }

  size_t i = 0;
  for (auto index: positions) {
    clusters[i++] = dataset[index];
  }
}

void seq_computer::update_cluster_positions() {
  int* count = new int[k];

  for (int j = 0; j < k; ++j) {
    clusters[j] = new Point();
  }

  for (int i = 0; i < n; ++i) {
    int cluster_for_point = cluster_for_point[i];
    count[cluster_for_point]++;
    clusters[cluster_for_point] += dataset[i];
  }

  for (int j = 0; j < k; ++j) {
    clusters[j] /= count[j];
  }
}

bool seq_computer::update_cluster_for_point() {
  bool change = false;

  for (int i = 0; i < n; i++) {
    Point datapoint = dataset[i];
    float min = distance_between(datapoint, ClusterPosition[0]);
    unsigned short index_min = 0;

    for (int j = 1; j < k; j++) {
      float distance = distance_between(datapoint, ClusterPosition[j]);
      if (distance < min) {
        min = distance;
        index_min = (unsigned short) j;
      }
    }

    if (cluster_for_point[i] != index_min) {
      cluster_for_point[i] = index_min;
      change = true;
    }

  }

  return change;
}
