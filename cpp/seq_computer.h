//
// Created by Paul Renauld on 2019-04-22.
//

#ifndef CPP_SEQ_COMPUTER_H
#define CPP_SEQ_COMPUTER_H


#include "kmean_computer.h"

class seq_computer: public kmean_computer {
  public:

    void init_starting_clusters() override ;
    // Returns true if something changed
    void update_cluster_positions() override;
    bool update_cluster_for_point() override;

};


#endif //CPP_SEQ_COMPUTER_H