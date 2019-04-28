//
// Created by Paul Renauld on 2019-04-22.
//

#ifndef CPP_SEQ_COMPUTER_H
#define CPP_SEQ_COMPUTER_H


#include "kmean_computer.h"
#include "data_structure.h"

class seq_computer : public kmean_computer {
  public:
    seq_computer(size_t k, size_t n, Dataset dataset) :
            kmean_computer(k, n, dataset) {}

    virtual ~seq_computer();

    float compute_silhouette() const override;

  protected:
    void init_starting_clusters() override;

    // Returns true if something changed
    void update_cluster_positions() override;

    bool update_cluster_for_point() override;

    void after_converge() override;

};


#endif //CPP_SEQ_COMPUTER_H
