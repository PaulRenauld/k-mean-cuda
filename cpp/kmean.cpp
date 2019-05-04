#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <getopt.h>
#include <algorithm>

#include "kmean_computer.h"
#include "Point.h"
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
          {"help", no_argument, 0,  '?'},
          {"output-file",   required_argument, 0,  'o'},
          {"time-file",   required_argument, 0,  'O'},
          {"input-file",    required_argument, 0,  'i'},
          {"cluster-count", required_argument, 0,  'k'},
          {"min",  required_argument, 0,  'm'},
          {"max",  required_argument, 0,  'M'},
          {"step", required_argument, 0,  's'},
          {"approximation", optional_argument, 0,  'a'},
          {0 ,0, 0, 0}
  };

  string output_file = "results", input_file = "random_points";
  string time_file_name = "";
  int k = -1;
  int min = 2, max = 100, step = 1;
  bool approx = false;

  while ((opt = getopt_long(argc, argv, "O:o:i:k:m:M:s:a?", long_options, NULL)) != EOF) {
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
      case 'm':
        min = (int) atoi(optarg);
        break;
      case 'M':
        max = (int) atoi(optarg);
        break;
      case 's':
        step = (int) atoi(optarg);
        break;
      case 'a':
        approx = true;
        break;
      case 'O':
        time_file_name = optarg;
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
    silhouette_finder finder(n, points, approx);
    Computer *best = nullptr;
    if (time_file_name.length() != 0) {
      ofstream time_file(time_file_name);
      if (time_file.is_open()) {
        best = finder.find_best_k(min, max, step, &time_file);
        time_file.close();
      } else cout << "Unable to open time file";
    } else {
      best = finder.find_best_k(min, max, step, &cout);
    }
    write_output_file(*best, output_file, file_args);
  }

  delete [] points;

  return 0;
}

void usage(char *string) {
  printf("%s computer\n", COMPUTER_TYPE);
  printf("Usage: %s [-o output_file] [-i input_file] [-k cluster_count/ -m min -M max -s step] [-a]\n", string);
  printf("Program Options:\n");
  printf("  -o  --output-file  <FILENAME>  Specify the output path for the cluster file\n");
  printf("  -i  --input-file  <FILENAME>   Name of the file to read the graph\n");
  printf("  -k  --cluster-count  <K>       Specify the number of cluster, if nothing is specified, this will try several k\n");
  printf("  -m  --min  <min>               Specify the minimum number of cluster to try when trying to find the best k (default 2) Doesn't apply when k is specified\n");
  printf("  -M  --max  <max>               Specify the maximum number of cluster to try when trying to find the best k (default 100) Doesn't apply when k is specified\n");
  printf("  -s  --step  <step>             Specify the step size when trying to find the best k (default 1) Doesn't apply when k is specified\n");
  printf("  -a  --approximation            Use an optimized approximation of the silhouette when trying to find the best k. Doesn't apply when k is specified\n");
  printf("  -O  --time-file <FILENAME>     If specified, write the time taken to find the best k in the file with other information in CSV\n");
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
