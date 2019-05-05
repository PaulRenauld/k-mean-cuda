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
inline void cudaAssert(cudaError_t code, const char *file, 
                       int line, bool abort=true) {
  if (code != cudaSuccess) {
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
#define POINT_PER_THREAD 128

__device__ float distance_square(Point first, Point second) {
  float diff_x = first.x - second.x;
  float diff_y = first.y - second.y;
  return diff_x * diff_x + diff_y * diff_y;
}

__global__ void kernel_update_cluster(GlobalConstants cuConstParams) {
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
    cuConstParams.cluster_for_point[index] = index_min;
    *cuConstParams.change = true;
  }
  return;
}

__global__ void kernel_update_cluster_accumulators(GlobalConstants cuConstParams) {
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

__global__ void kernel_update_cluster_positions(GlobalConstants cuConstParams) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= cuConstParams.k) return;
  ClusterAccumulator acc = cuConstParams.accumulators[k];
  cuConstParams.clusters[k].x = acc.x / acc.count;
  cuConstParams.clusters[k].y = acc.y / acc.count;
  cuConstParams.accumulators[k].x = 0;
  cuConstParams.accumulators[k].y = 0;
  cuConstParams.accumulators[k].count = 0;
}


__global__ void kernel_silhouette_all(GlobalConstants cuConstParams, float *avg, float *glob_dist, unsigned int *glob_count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= cuConstParams.n) return;

  const size_t n  = cuConstParams.n;
  const size_t k  = cuConstParams.k;

  float *mean_to_clust = glob_dist + i * k;
  unsigned int *cluster_count = glob_count + i * k;

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

__global__ void kernel_silhouette_approx(GlobalConstants cuConstParams, float *avg) {
  __shared__ float shared_avg;
  if (threadIdx.x == 0) {
    shared_avg = 0;
  }
  __syncthreads();
  
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= cuConstParams.n) return;

  unsigned short cluster = cuConstParams.cluster_for_point[i];
  Point point = cuConstParams.dataset[i];

  float b_i = HUGE_VALF;
  float a_i = sqrtf(distance_square(point, cuConstParams.clusters[cluster]));

  for (size_t c = 0; c < cuConstParams.k; ++c) {
    if (c == cluster) continue;
    float dist = sqrtf(distance_square(point, cuConstParams.clusters[c]));
    if (dist < b_i) {
      b_i = dist;
    }
  }

  float denom = max(a_i, b_i);
  float s = (b_i - a_i) / denom;
  
  atomicAdd(&shared_avg, s);

  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(avg, shared_avg);
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

par_computer::par_computer(size_t k, size_t n, Dataset dataset) : 
      kmean_computer(k, n, dataset)
       {
  cudaCheckError( cudaMalloc(&cudaDeviceDataset, sizeof(Point) * n) );
  cudaCheckError( cudaMemcpy(cudaDeviceDataset, dataset, sizeof(Point) * n, cudaMemcpyHostToDevice) );

  cudaCheckError( cudaMalloc(&cudaDeviceClusters, sizeof(Point) * k) );

  cudaCheckError( cudaMalloc(&cuda_device_cluster_for_point, sizeof(unsigned short) * n) );

  cudaCheckError( cudaMalloc(&clusterAccumulators, sizeof(ClusterAccumulator) * k) );

  cudaCheckError( cudaMalloc(&change, sizeof(bool)) );

  cuConstParams.k = k;
  cuConstParams.n = n;
  cuConstParams.dataset = cudaDeviceDataset;
  cuConstParams.clusters = cudaDeviceClusters;
  cuConstParams.cluster_for_point = cuda_device_cluster_for_point;
  cuConstParams.accumulators = clusterAccumulators;
  cuConstParams.change = change;

  cudaCheckError( cudaDeviceSynchronize() );

  block_dim_points = dim3(256, 1); 
  grid_dim_points = dim3((n + block_dim_points.x - 1) / block_dim_points.x, 1);
  block_dim_clusters = dim3(256, 1);
  grid_dim_clusters = dim3((k + block_dim_clusters.x - 1) / block_dim_clusters.x, 1);
}

par_computer::~par_computer() {
  cudaFree(cudaDeviceDataset);
  cudaFree(cudaDeviceClusters);
  cudaFree(cuda_device_cluster_for_point);
  cudaFree(clusterAccumulators);
  cudaFree(change);
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
  kernel_update_cluster_positions<<<grid_dim_clusters, block_dim_clusters>>>(cuConstParams);

  cudaCheckError( cudaMemcpy(cudaDeviceClusters, clusters, sizeof(Point) * k, cudaMemcpyHostToDevice) );
}

void par_computer::update_cluster_positions() {
  dim3 blockDim(256, 1);
  dim3 gridDim(n / (blockDim.x + POINT_PER_THREAD) + 1);
  kernel_update_cluster_accumulators<<<gridDim, blockDim, k * sizeof(ClusterAccumulator)>>>(cuConstParams);
  cudaCheckKernelError();

  kernel_update_cluster_positions<<<grid_dim_clusters, block_dim_clusters>>>(cuConstParams);
  cudaCheckKernelError();
}

bool par_computer::update_cluster_for_point() {
  bool changeHost = false;
  cudaCheckError( cudaMemcpy(change, &changeHost, sizeof(bool), cudaMemcpyHostToDevice) );
  kernel_update_cluster<<<grid_dim_points, block_dim_points>>>(cuConstParams);
  cudaCheckKernelError();

  cudaCheckError( cudaMemcpy(&changeHost, change, sizeof(bool), cudaMemcpyDeviceToHost) );

  return changeHost;
}


void par_computer::after_converge() {
  cudaCheckError( cudaMemcpy(clusters, cudaDeviceClusters, sizeof(Point) * k, cudaMemcpyDeviceToHost) );
  cudaCheckError( cudaMemcpy(cluster_for_point, cuda_device_cluster_for_point, sizeof(unsigned short) * n, cudaMemcpyDeviceToHost) );
}


float par_computer::compute_silhouette_all() const {
  return compute_silhouette(false);
}


float par_computer::compute_silhouette_approximation() const {
  return compute_silhouette(true);
}

float par_computer::compute_silhouette(bool approx) const {
  float avg = 0;
  float *avg_ptr;
  cudaCheckError( cudaMalloc(&avg_ptr, sizeof(float)) );
  cudaCheckError( cudaMemcpy(avg_ptr, &avg, sizeof(float), cudaMemcpyHostToDevice) );

  dim3 blockDim(256, 1);
  dim3 gridDim((n + blockDim.x - 1) / blockDim.x, 1);
  if (approx) {
    kernel_silhouette_approx<<<gridDim, blockDim>>>(cuConstParams, avg_ptr);
    cudaCheckKernelError();
  } else {
    float *glob_dist;
    cudaCheckError( cudaMalloc(&glob_dist, sizeof(float) * n * k) );

    unsigned int *glob_count;
    cudaCheckError( cudaMalloc(&glob_count, sizeof(unsigned int) * n * k) );

    kernel_silhouette_all<<<gridDim, blockDim>>>(cuConstParams,avg_ptr, glob_dist, glob_count);
    cudaCheckKernelError();

    cudaCheckError( cudaFree(glob_dist) );
    cudaCheckError( cudaFree(glob_count) );
  }

  cudaCheckError( cudaMemcpy(&avg, avg_ptr, sizeof(float), cudaMemcpyDeviceToHost) );
  cudaCheckError( cudaFree(avg_ptr) );
  
  avg /= n;
  return avg;
}