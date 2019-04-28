#include "par_computer.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <iostream>
#include <set>
#include <iterator>
#include <random>

#define DEBUG

#ifdef DEBUG
#define cudaCheckError(ans)  cudaAssert((ans), __FILE__, __LINE__);
inline void cudaAssert(cudaError_t code, const char
*
file, int line, bool abort=true)
{
if (code != cudaSuccess)
{
fprintf(stderr, "CUDA Error: %s at %s:%d\n",
cudaGetErrorString(code), file, line);
if (abort) exit(code);
}
}
#else
#define cudaCheckError(ans) ans
#endif

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

__device__ float distance_square(Point first, Point second) {
  float diff_x = first.x - second.x;
  float diff_y = first.y - second.y;
  return diff_x * diff_x + diff_y * diff_y;
}

__global__ void kernel_update_cluster(bool* change) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  printf("THE FUCK\n");

  if (index >= cuConstParams.n) return;

  Point datapoint = cuConstParams.dataset[index];
  // *((int*)NULL) = 15;
  Point f = cuConstParams.clusters[0];
  float minimum = datapoint.distance_squared_to(f);
  // float minimum = distance_square(datapoint, f);
  unsigned short index_min = 0;

  for (unsigned short j = 1; j < cuConstParams.k; j++) {
  //   float distance = datapoint.distance_squared_to(cuConstParams.clusters[j]);
  //   if (distance < minimum) {
  //     minimum = distance;
  //     index_min = j;
  //   }
  }
  // printf("k=%zu\n", cuConstParams.k);
  // if (cuConstParams.cluster_for_point[index] != index_min) {
  //   printf("New cluster for point: %u \n", index_min);
  //   cuConstParams.cluster_for_point[index] = index_min;
  //   *change = true;
  // }
  printf("THE FUCK3333\n");
  return;
}

__global__ void kernel_update_cluster_positions() {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  printf("THE FUCK2\n");
  // *((int*)NULL) = 15;

  if (index >= cuConstParams.k) return;

  // int count = 0;
  // Point position;

  // for (int i = 0; i < cuConstParams.n; ++i) {
  //   if (cuConstParams.cluster_for_point[i] == index) {
  //     count++;
  //     position += cuConstParams.dataset[i];
  //   }
  // }

  // position /= count;
  // cuConstParams.clusters[index] = position;
  return;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

par_computer::par_computer(size_t k, size_t n, Dataset dataset) : kmean_computer(k, n, dataset) {
  init_starting_clusters(); 

  cudaCheckError( cudaMalloc(&cudaDeviceDataset, sizeof(Point) * n) );
  cudaCheckError( cudaMemcpy(cudaDeviceDataset, dataset, sizeof(Point) * n, cudaMemcpyHostToDevice) );

  cudaCheckError( cudaMalloc(&cudaDeviceClusters, sizeof(Point) * k) );
  cudaCheckError( cudaMemcpy(cudaDeviceClusters, clusters, sizeof(Point) * k, cudaMemcpyHostToDevice) );

  cudaCheckError( cudaMalloc(&cuda_device_cluster_for_point, sizeof(unsigned short) * n) );

  GlobalConstants params;
  params.k = k;
  params.n = n;
  params.dataset = cudaDeviceDataset;
  params.clusters = cudaDeviceClusters;
  params.cluster_for_point = cuda_device_cluster_for_point;

  cudaCheckError( cudaMemcpyToSymbol(cuConstParams, &params, sizeof(GlobalConstants)) );

  cudaCheckError( cudaDeviceSynchronize() );
}

par_computer::~par_computer() {
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
  dim3 blockDim(256, 1);
  dim3 gridDim((k + blockDim.x - 1) / blockDim.x);

  kernel_update_cluster_positions<<<gridDim, blockDim>>>();
  cudaCheckError( cudaDeviceSynchronize() );
}

bool par_computer::update_cluster_for_point() {
  dim3 blockDim(256, 1);
  dim3 gridDim(1, 1);

  bool* change;
  cudaCheckError( cudaMalloc(&change, sizeof(bool)) );

  kernel_update_cluster<<<gridDim, blockDim>>>(change);
  cudaCheckError( cudaDeviceSynchronize() );

  bool changeHost = false;
  cudaCheckError( cudaMemcpy(&changeHost, change, sizeof(bool), cudaMemcpyDeviceToHost) );

  return changeHost;
}

ClusterPosition par_computer::converge() {

  std::cout << "Converge" << n << " " << k << std::endl;
  while (update_cluster_for_point()) {
    std::cout << "One iteration" << std::endl;
    update_cluster_positions();
  }
    update_cluster_positions();

  cudaCheckError( cudaMemcpy(clusters, cudaDeviceClusters, sizeof(Point) * k, cudaMemcpyDeviceToHost) );
  cudaCheckError( cudaMemcpy(cluster_for_point, cuda_device_cluster_for_point, sizeof(unsigned short) * n, cudaMemcpyDeviceToHost) );

  return clusters;
}