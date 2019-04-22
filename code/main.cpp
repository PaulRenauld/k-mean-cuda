#include <getopt.h>

#include "main.h"

int main (int argc, char *argv[]) {
    int k = 5;
    char *optstring = "o:i:k:";
    FILE input = NULL;
    FILE output = NULL;

    while ((c = getopt(argc, argv, optstring)) != -1) {
        switch (c) {
            case 'o':
                output = fopen(optarg, "w");
                break;
            case 'i':
                input = fopen(optarg, "r");
                if (input == NULL) {
                    exit(1);
                }
                break;
            case 'k':
                k = (int) optarg;
                break;
            default:
                exit(1);
        }
    }

    Cluster clusters [k];
    initalize_clusters(clusters, k);

    main_loop(clusters, dataset);

    write_output(clusters, output);
}

class Point {
    float x;
    float y;

};

class Cluster {
    Point position;
    int count;
};