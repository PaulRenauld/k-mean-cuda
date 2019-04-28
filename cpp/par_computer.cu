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
#define cudaCheckKernelError() cudaCheckError( cudaDeviceSynchronize () )
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
#define cudaCheckKernelError() 
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
  ClusterAccumulator *accumulators;
  unsigned short *cluster_for_point;
};

#define POINT_PER_THREAD 128

__constant__ GlobalConstants cuConstParams;

__device__ float distance_square(Point first, Point second) {
  float diff_x = first.x - second.x;
  float diff_y = first.y - second.y;
  return diff_x * diff_x + diff_y * diff_y;
}

__global__ void kernel_update_cluster(bool* change) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index >= cuConstParams.n) return;

  Point datapoint = cuConstParams.dataset[index];
  Point f = cuConstParams.clusters[0];
  // float minimum = datapoint.distance_squared_to(f);
  float minimum = distance_square(datapoint, f);
  unsigned short index_min = 0;

  for (unsigned short j = 1; j < cuConstParams.k; j++) {
    float distance = distance_square(datapoint, cuConstParams.clusters[j]);
    if (distance < minimum) {
      minimum = distance;
      index_min = j;
    }
  }

  if (cuConstParams.cluster_for_point[index] != index_min) {
    // printf("New cluster for point: %u \n", index_min);
    cuConstParams.cluster_for_point[index] = index_min;
    *change = true;
  }
  return;
}

__global__ void kernel_update_cluster_accumulators() {
  extern __shared__ ClusterAccumulator accs[];
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  for (int k = threadIdx.x; k < cuConstParams.k; k += blockDim.x) {
    accs[k].x = 0;
    accs[k].y = 0;
    accs[k].count = 0;
  }
  __syncthreads();

  size_t max = (index + 1) * POINT_PER_THREAD;
  max = max > cuConstParams.n ? cuConstParams.n : max;
  
  for (size_t i = index * POINT_PER_THREAD; i < max; ++i) {
    Point point = cuConstParams.dataset[i];
    size_t cluster = cuConstParams.cluster_for_point[i];
    atomicAdd(&accs[cluster].x, point.x);
    atomicAdd(&accs[cluster].y, point.y);
    atomicAdd(&accs[cluster].count, 1);
  }

  __syncthreads();
  for (int k = threadIdx.x; k < cuConstParams.k; k += blockDim.x) {
    atomicAdd(&cuConstParams.accumulators[k].x, accs[k].x);
    atomicAdd(&cuConstParams.accumulators[k].y, accs[k].y);
    atomicAdd(&cuConstParams.accumulators[k].count, accs[k].count);
  }
}

__global__ void kernel_update_cluster_positions() {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= cuConstParams.k) return;
  ClusterAccumulator acc = cuConstParams.accumulators[k];
  cuConstParams.clusters[k].x = acc.x / acc.count;
  cuConstParams.clusters[k].y = acc.y / acc.count;
  cuConstParams.accumulators[k].x = 0;
  cuConstParams.accumulators[k].y = 0;
  cuConstParams.accumulators[k].count = 0;
}

__global__ void kernel_silhouette(float *avg, float *glob_dist, unsigned int *glob_count, unsigned int offset) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= cuConstParams.n) return;

  const size_t n  = cuConstParams.n;
  const size_t k  = cuConstParams.k;

  float *mean_to_clust = glob_dist + i * offset;
  unsigned int *cluster_count = glob_count + i * offset;

  unsigned short cluster = cuConstParams.cluster_for_point[i];
  Point point = cuConstParams.dataset[i];

  for (size_t a = 0; a < k; ++a) {
    mean_to_clust[a] = 0.f;
    cluster_count[a] = 0;
  }

  for (size_t j = 0; j < n; ++j) {
    unsigned short cluster_j = cuConstParams.cluster_for_point[j];
    float distance = sqrtf(distance_square(point, cuConstParams.dataset[j]));

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

  float denom = max(a_i, b_i);
  float s = (b_i - a_i) / denom;
  atomicAdd(avg, s);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
inline unsigned int next_multiple_of(unsigned int x, unsigned int mod) {
  return mod * (x / mod + 1);
}

par_computer::par_computer(size_t k, size_t n, Dataset dataset) : kmean_computer(k, n, dataset) {
  cudaCheckError( cudaMalloc(&cudaDeviceDataset, sizeof(Point) * n) );
  cudaCheckError( cudaMemcpy(cudaDeviceDataset, dataset, sizeof(Point) * n, cudaMemcpyHostToDevice) );

  cudaCheckError( cudaMalloc(&cudaDeviceClusters, sizeof(Point) * k) );

  cudaCheckError( cudaMalloc(&cuda_device_cluster_for_point, sizeof(unsigned short) * n) );

  cudaCheckError( cudaMalloc(&clusterAccumulators, sizeof(ClusterAccumulator) * k) );

  GlobalConstants params;
  params.k = k;
  params.n = n;
  params.dataset = cudaDeviceDataset;
  params.clusters = cudaDeviceClusters;
  params.cluster_for_point = cuda_device_cluster_for_point;
  params.accumulators = clusterAccumulators;

  cudaCheckError( cudaMemcpyToSymbol(cuConstParams, &params, sizeof(GlobalConstants)) );

  cudaCheckError( cudaDeviceSynchronize() );
}

par_computer::~par_computer() {
  cudaFree(cudaDeviceDataset);
  cudaFree(cudaDeviceClusters);
  cudaFree(cuda_device_cluster_for_point);
  cudaFree(clusterAccumulators);
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

  // Clear the accumulators
  dim3 blockDim(256, 1);
  dim3 gridDim((k + blockDim.x - 1) / blockDim.x);
  kernel_update_cluster_positions<<<gridDim, blockDim>>>();

  cudaCheckError( cudaMemcpy(cudaDeviceClusters, clusters, sizeof(Point) * k, cudaMemcpyHostToDevice) );
}

void par_computer::update_cluster_positions() {
  dim3 blockDim(256, 1);
  dim3 gridDim(n / (blockDim.x + POINT_PER_THREAD) + 1);
  kernel_update_cluster_accumulators<<<gridDim, blockDim, k * sizeof(ClusterAccumulator)>>>();
  cudaCheckKernelError();

  gridDim = dim3((k + blockDim.x - 1) / blockDim.x);
  kernel_update_cluster_positions<<<gridDim, blockDim>>>();
  cudaCheckKernelError();
}

bool par_computer::update_cluster_for_point() {
  dim3 blockDim(512, 1);
  dim3 gridDim((n + blockDim.x - 1) / blockDim.x, 1);

  bool* change;
  cudaCheckError( cudaMalloc(&change, sizeof(bool)) );

  kernel_update_cluster<<<gridDim, blockDim>>>(change);
  cudaCheckKernelError();

  bool changeHost = false;
  cudaCheckError( cudaMemcpy(&changeHost, change, sizeof(bool), cudaMemcpyDeviceToHost) );

  // cudaCheckError( cudaFree(change) );

  return changeHost;
}


void par_computer::after_converge() {
  cudaCheckError( cudaMemcpy(clusters, cudaDeviceClusters, sizeof(Point) * k, cudaMemcpyDeviceToHost) );
  cudaCheckError( cudaMemcpy(cluster_for_point, cuda_device_cluster_for_point, sizeof(unsigned short) * n, cudaMemcpyDeviceToHost) );
}


float par_computer::compute_silhouette() const {
  float* avg_ptr;
  float avg = 0;
  cudaCheckError( cudaMalloc(&avg_ptr, sizeof(float)) );
  cudaCheckError( cudaMemcpy(avg_ptr, &avg, sizeof(float), cudaMemcpyHostToDevice) );

  unsigned int offset = next_multiple_of(k * sizeof(float), 128);

  float *glob_dist;
  cudaCheckError( cudaMalloc(&glob_dist, sizeof(float) * n * offset) );

  unsigned int *glob_count;
  cudaCheckError( cudaMalloc(&glob_count, sizeof(unsigned int) * n * offset) );

  dim3 blockDim(256, 1);
  dim3 gridDim((n + blockDim.x - 1) / blockDim.x, 1);
  kernel_silhouette<<<gridDim, blockDim>>>(avg_ptr, glob_dist, glob_count, offset);
  cudaCheckKernelError();

  cudaCheckError( cudaFree(glob_dist) );
  cudaCheckError( cudaFree(glob_count) );

  cudaCheckError( cudaMemcpy(&avg, avg_ptr, sizeof(float), cudaMemcpyDeviceToHost) );
  // cudaCheckError( cudaFree(avg_ptr) );
  
  avg /= n;
  return avg;
}
