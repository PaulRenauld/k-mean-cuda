#include <iostream>
#include <fstream>
#include <string>

#include "cxxopts.hpp"
#include "kmean_computer.h"
#include "Point.h"
#include "seq_computer.h"

using namespace std;

typedef seq_computer Computer;

Computer
parse_input(const cxxopts::ParseResult &arguments, vector<string> &file_args);

void write_output_file(const Computer &computer,
                       const cxxopts::ParseResult &arguments,
                       const vector<string> &file_args);

bool contains(const string &str, const string &beginning) {
  return str.find(beginning, 0) != string::npos;
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
           cxxopts::value<size_t>()->default_value("5"));

  auto arguments = options.parse(argc, argv);

  vector<string> file_args;
  Computer computer = parse_input(arguments, file_args);
  computer.converge();
  write_output_file(computer, arguments, file_args);

  return 0;
}

void write_output_file(const Computer &computer,
                       const cxxopts::ParseResult &arguments,
                       const vector<string> &file_args) {
  const string output_file_name = arguments["output-file"].as<string>();
  ofstream output_file(output_file_name);
  if (output_file.is_open()) {
    for (const auto& arg: file_args) {
      output_file << arg << std::endl;
    }
    output_file << computer;
    output_file.close();
  } else cout << "Unable to open file";
}

Computer
parse_input(const cxxopts::ParseResult &arguments, vector<string> &file_args) {
  string input_file_name = arguments["input-file"].as<string>();

  string line;
  Point *points = nullptr;
  size_t k = arguments["cluster-count"].as<size_t>();
  size_t n = 0;
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
        file_args.push_back(line);
      } else if (contains(line, "width") || contains(line, "height") ||
                 contains(line, "dim")) {
        cout << "Parameter " << line << endl;
        file_args.push_back(line);
      } else if (!contains(line, "C")) {
        if (i >= n) {
          cout << "Parsing error: too many points" << endl;
        } else {
          points[i++] = Point(line);
        }
      }
    }
    input_file.close();
  } else cout << "Unable to open file";
  return {k, n, points};
}
