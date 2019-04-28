//
// Created by math on 4/27/2019.
//

#ifndef K_MEAN_CUDA_PAR_COMPUTER_H
#define K_MEAN_CUDA_PAR_COMPUTER_H

#include "Point.h"
#include "data_structure.h"
#include "kmean_computer.h"
struct ClusterAccumulator {
      float x;
      float y;
      unsigned int count;
    };

class par_computer : public kmean_computer {
  public:
    par_computer(size_t k, size_t n, Dataset dataset);
    ~par_computer();

  private:

    Dataset cudaDeviceDataset;
    ClusterPosition cudaDeviceClusters;
    unsigned short *cuda_device_cluster_for_point;
    ClusterAccumulator *clusterAccumulators;


  protected:
    void init_starting_clusters() override;

    // Returns true if something changed
    void update_cluster_positions() override;

    bool update_cluster_for_point() override;

    void after_converge() override;
};


#endif //K_MEAN_CUDA_PAR_COMPUTER_H
