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

}

bool seq_computer::update_cluster_for_point() {
  return false;
}
