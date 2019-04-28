#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <getopt.h>
#include <algorithm>

#include "kmean_computer.h"
#include "Point.h"
// #include "seq_computer.h"
#include "silhouette_finder.h"


using namespace std;

Dataset parse_input(string input_file_name,
                    vector<string> &file_args, size_t *n);

void write_output_file(const Computer &computer,
                       const string output_file_name,
                       const vector<string> &file_args);

void usage(char *string);

bool contains(const string &str, const string &beginning) {
  return str.find(beginning, 0) != string::npos;
}

void delete_whitespaces(string &str) {
  str.erase(remove_if(str.begin(), str.end(), ::isspace), str.end());
}

int main(int argc, char *argv[]) {
  int opt;
  static struct option long_options[] = {
          {"help",     no_argument, 0,  '?'},
          {"output-file",    required_argument, 0,  'o'},
          {"input-file",    required_argument, 0,  'i'},
          {"cluster-count",     required_argument, 0,  'k'},
          {0 ,0, 0, 0}
  };

  string output_file = "results", input_file = "random_points";
  int k = -1;

  while ((opt = getopt_long(argc, argv, "o:i:k:?", long_options, NULL)) != EOF) {
    switch (opt) {
      case 'o':
        output_file = optarg;
        break;
      case 'i':
        input_file = optarg;
        break;
      case 'k':
        k = (int) atoi(optarg);
        break;
      default:
        usage(argv[0]);
        exit(1);
    }
  }

  vector<string> file_args;
  size_t n = 0;
  Dataset points = parse_input(input_file, file_args, &n);

  if (k > 0) {
    Computer computer(k, n, points);
    computer.converge();
    write_output_file(computer, output_file, file_args);
  } else {
    silhouette_finder finder(n, points);
    Computer* best = finder.find_best_k(2, 40, &cout);
    write_output_file(*best, output_file, file_args);
  }

  delete [] points;

  return 0;
}

void usage(char *string) {
  printf("%s computer\n", COMPUTER_TYPE);
  printf("Usage: %s [-o output_file] [-i input_file] [-k cluster_count]\n", string);
  printf("Program Options:\n");
  printf("  -o  --output-file  <FILENAME>  Specify the output path for the cluster file\n");
  printf("  -i  --input-file  <FILENAME>   Name of the file to read the graph\n");
  printf("  -k  --cluster-count  <K>       Specify the number of cluster, if nothing is specified, this will try several k\n");
  printf("  -?  --help                     This message\n");
}

void write_output_file(const Computer &computer,
                       const string output_file_name,
                       const vector<string> &file_args) {
  ofstream output_file(output_file_name);
  if (output_file.is_open()) {
    for (const auto &arg: file_args) {
      output_file << arg << std::endl;
    }
    output_file << computer;
    output_file.close();
  } else cout << "Unable to open file";
}

Dataset parse_input(string input_file_name,
                    vector<string> &file_args, size_t *n) {

  string line;
  Point *points = nullptr;
  *n = 0;
  size_t i = 0;
  ifstream input_file(input_file_name);
  if (input_file.is_open()) {
    while (getline(input_file, line)) {
      delete_whitespaces(line);
      if (contains(line, "point-count")) {
        *n = stoi(line.substr(line.rfind(',') + 1));
        delete[] points;
        points = new Point[*n];
        i = 0;
        file_args.push_back(line);
      } else if (contains(line, "width") || contains(line, "height") ||
                 contains(line, "dim")) {
        cout << "Parameter " << line << endl;
        file_args.push_back(line);
      } else if (!contains(line, "C")) {
        if (i >= *n) {
          cout << "Parsing error: too many points" << endl;
        } else {
          points[i++] = Point(line);
        }
      }
    }
    input_file.close();
  } else cout << "Unable to open file";
  return points;
}
