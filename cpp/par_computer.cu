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

__device__ __inline__ float distance_square(Point first, Point second) {
  float diff_x = first.x - second.x;
  float diff_y = first.y - second.y;
  return diff_x * diff_x + diff_y * diff_y;
}

__global__ void kernel_update_cluster(bool* change) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  Point& datapoint = cuConstParams.dataset[index];
  float min = distance_square(datapoint, cuConstParams.clusters[0]);
  unsigned short index_min = 0;

  for (unsigned short j = 1; j < cuConstParams.k; j++) {
    float distance = distance_square(datapoint, cuConstParams.clusters[j]);
    if (distance < min) {
      min = distance;
      index_min = j;
    }
  }

  if (cuConstParams.cluster_for_point[index] != index_min) {
    cuConstParams.cluster_for_point[index] = index_min;
    *change = true;
  }
}

__global__ void kernel_update_cluster_positions() {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int count = 0;
  Point position;

  for (int i = 0; i < cuConstParams.n; ++i) {
    if (cuConstParams.cluster_for_point[i] == index) {
      count++;
      position += cuConstParams.cluster_for_point[i];
    }
  }

  position /= count;
  cuConstParams.clusters[index] = position;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

void par_computer::par_computer(size_t k, size_t n, Dataset dataset) k(k), n(n), dataset(dataset){
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

void par_computer::~par_computer() {
  delete[] clusters;
  delete[] cluster_for_point;

  cudaFree(cudaDeviceDataset);
  cudaFree(cudaDeviceClusters);
  cudaFree(cuda_device_cluster_for_point);
}

void par_computer::init_starting_clusters() {
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

void par_computer::update_cluster_positions() {
  dim3 blockDim(256, 1);
  dim3 gridDim((k + blockDim.x - 1) / blockDim.x);

  kernel_update_cluster_positions<<<gridDim, blockDim>>>();
}

bool par_computer::update_cluster_for_point() {
  dim3 blockDim(256, 1);
  dim3 gridDim((n + blockDim.x - 1) / blockDim.x);

  bool* change;
  cudaMalloc(&change, sizeof(bool));

  kernel_update_cluster<<<gridDim, blockDim>>>(change);

  bool changeHost;
  cudaMemcpy(&changeHost, change, cudaMemcpyDeviceToHost);

  return changeHost;
}

ClusterPosition par_computer::converge() {
  while (update_cluster_for_point()) {
    update_cluster_positions();
  }

  cudaMemcpy(clusters, cudaDeviceClusters, sizeof(Point) * k, cudaMemcpyDeviceToHost);

  return clusters;
}