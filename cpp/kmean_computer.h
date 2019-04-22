//
// Created by Paul Renauld on 2019-04-22.
//

#ifndef CPP_KMEAN_COMPUTER_H
#define CPP_KMEAN_COMPUTER_H

#include <vector>
#include <ostream>
#include "Point.h"
#include "data_structure.h"

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

    float compute_silhouette() const {
      float avg = 0.0;
      for (size_t i = 0; i < n; ++i) {
        unsigned short cluster = cluster_for_point[i];
        Point& point = dataset[i];
        float a[k];
        size_t a_count[k];

        for (size_t j = 0; j < n; ++j) {
          unsigned short cluster_j = cluster_for_point[j];
          float distance = sqrt(point.distance_squared_to(dataset[j]));

          a[cluster_j] += distance;
          a_count[cluster_j]++;
        }

        a_count[cluster]--;

        float b = ;
        float a_i = 0.0;

        for (int c = 0; c < k; ++c) {
          a[c] /= a_count[c];

          if (c == cluster) {
            a_i = a[c];
          }
          else if (a[c] < b) {
            b = a[c];
          }
        }
      }

      return avg;
    }

    friend std::ostream &
    operator<<(std::ostream &os, const kmean_computer &computer) {
      for (size_t curr_k = 0; curr_k < computer.k; ++curr_k) {
        os << "C," << computer.clusters[curr_k] << std::endl;
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

