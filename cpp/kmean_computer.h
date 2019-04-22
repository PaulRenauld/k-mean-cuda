//
// Created by Paul Renauld on 2019-04-22.
//

#ifndef CPP_KMEAN_COMPUTER_H
#define CPP_KMEAN_COMPUTER_H

#include <vector>
#include <ostream>
#include "Point.h"

typedef Point *Dataset;

typedef Point *ClusterPosition;

class kmean_computer {
  public:
    kmean_computer(size_t k, size_t n, Dataset dataset) : k(k), n(n),
                                                          dataset(dataset) {
      clusters = new Point[k];
      cluster_for_point = new unsigned short[n];
    }

    ~kmean_computer() {
      delete[] clusters;
      delete[] cluster_for_point;
    }

    ClusterPosition converge() {
      init_starting_clusters();
      while (update_cluster_for_point()) {
        update_cluster_positions();
      }
      return clusters;
    }

    friend std::ostream &
    operator<<(std::ostream &os, const kmean_computer &computer) {
      for (size_t curr_k = 0; curr_k < computer.k; ++curr_k) {
        os << "C," << computer.clusters[curr_k];
        for (int i = 0; i < computer.n; ++i) {
          if (computer.cluster_for_point[i] == curr_k) {
            os << computer.dataset[i] << std::endl;
          }
        }
      }
      return os;
    }

    const size_t k;
    const size_t n;

  protected:
    const Dataset dataset;
    ClusterPosition clusters;
    unsigned short *cluster_for_point;

    virtual void init_starting_clusters() = 0;

    // Returns true if something changed
    virtual void update_cluster_positions() = 0;

    virtual bool update_cluster_for_point() = 0;
};

#endif //CPP_KMEAN_COMPUTER_H

