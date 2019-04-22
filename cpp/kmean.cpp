#include <iostream>
#include <fstream>
#include <string>

#include "cxxopts.hpp"
#include "kmean_computer.h"
#include "Point.h"

using namespace std;

bool contains(const string &str, const string &beginning) {
  return str.find(str, 0) != string::npos;
}

void delete_whitespaces(string &str) {
  str.erase(remove_if(str.begin(), str.end(), ::isspace), str.end());
}


int main(int argc, char *argv[]) {
  cxxopts::Options options("Sequential K-mean",
                           "Divide the dataset into k clusters");

  options.add_options()
          ("o,output-file", "Name of the file to store the generated arguments",
           cxxopts::value<std::string>()->default_value("results"))
          ("i,input-file", "Name of the file to read the graph",
           cxxopts::value<std::string>()->default_value("random_points"))
          ("k,cluster-count", "Specify the number of cluster",
           cxxopts::value<int>()->default_value("5"));

  auto arguments = options.parse(argc, argv);

  std::cout << "Hello, World!" << arguments["output-file"].as<string>()
            << arguments["input-file"].as<string>()
            << arguments["cluster-count"].as<int>() << std::endl;


  string input_file_name = arguments["input-file"].as<string>();
  vector<string> file_args;

  string line;
  Point *points = nullptr;
  size_t n;
  size_t i = 0;
  ifstream input_file(input_file_name);
  if (input_file.is_open()) {
    while (getline(input_file, line)) {
      delete_whitespaces(line);
      if (contains(line, "point-count")) {
        n = stoi(line.substr(line.rfind(',') + 1));
        delete[] points;
        points = new Point[n];
        i = 0;
      } else if (!contains(line, "width") && !contains(line, "height") &&
                 !contains(line, "dim") && !contains(line, "C")) {
        points[i++] = Point(line);
      }
    }
    input_file.close();
  } else cout << "Unable to open file";
  return 0;
}
