#ifndef CPP_KMEAN_COMPUTER_H
#define CPP_KMEAN_COMPUTER_H

#include <vector>
#include <math.h>
#include <ostream>
#include "Point.h"
#include "data_structure.h"

class kmean_computer {
  public:
    kmean_computer(size_t k, size_t n, Dataset dataset);

    ~kmean_computer();

    ClusterPosition converge();

    virtual float compute_silhouette() const = 0;

    friend std::ostream &operator<<(std::ostream &os,
                                    const kmean_computer &computer);

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

    // Action that should be done after converge is executed
    virtual void after_converge() = 0;
};

#endif //CPP_KMEAN_COMPUTER_H

