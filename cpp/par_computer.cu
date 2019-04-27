#include "par_computer.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <set>
#include <iterator>
#include <random>

////////////////////////////////////////////////////////////////////////////////////////
// All cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////
struct GlobalConstants {
  size_t k;
  size_t n;
  Dataset dataset;
  ClusterPosition clusters;
  unsigned short *cluster_for_point;
};

__constant__ GlobalConstants cuConstParams;

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

par_computer::par_computer(size_t k, size_t n, Dataset dataset) : kmean_computer(k, n, dataset) {
  clusters = new Point[k];
  cluster_for_point = new unsigned short[n];

  init_starting_clusters(); 

  cudaMalloc(&cudaDeviceDataset, sizeof(Point) * n);
  cudaMemcpy(cudaDeviceDataset, dataset, sizeof(Point) * n, cudaMemcpyHostToDevice);

  cudaMalloc(&cudaDeviceClusters, sizeof(Point) * k);
  cudaMemcpy(cudaDeviceClusters, clusters, sizeof(Point) * k, cudaMemcpyHostToDevice);

  cudaMalloc(&cuda_device_cluster_for_point, sizeof(unsigned short) * n);

  GlobalConstants params;
  params.k = k;
  params.n = n;
  params.dataset = cudaDeviceDataset;
  params.clusters = cudaDeviceClusters;
  params.cluster_for_point = cuda_device_cluster_for_point;

  cudaMemcpyToSymbol(cuConstParams, &params, sizeof(GlobalConstants));

}

par_computer::~par_computer() {
  delete[] clusters;
  delete[] cluster_for_point;

  cudaFree(cudaDeviceDataset);
  cudaFree(cudaDeviceClusters);
  cudaFree(cuda_device_cluster_for_point);
}

void par_computer::init_starting_clusters() {
  std::default_random_engine generator;
  std::uniform_int_distribution<size_t> distribution(0, n - 1);

  std::set<size_t, std::greater<size_t>> positions;
  while (positions.size() < k) {
    positions.insert(distribution(generator));
  }

  size_t i = 0;
  for (auto index: positions) {
    clusters[i++] = dataset[index];
  }
}

void par_computer::update_cluster_positions() {

}

bool par_computer::update_cluster_for_point() {

}

ClusterPosition par_computer::converge() {
  while (update_cluster_for_point()) {
    update_cluster_positions();
  }

  cudaMemcpy(clusters, cudaDeviceClusters, sizeof(Point) * k, cudaMemcpyDeviceToHost);

  return clusters;
}