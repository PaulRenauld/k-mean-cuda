//
// Created by Paul Renauld on 2019-04-17.
//

#include "Point.h"
#include <string>

__device__ __host__ bool Point::operator==(const Point &rhs) const {
  return x == rhs.x &&
         y == rhs.y;
}

__device__ __host__ bool Point::operator!=(const Point &rhs) const {
  return !(rhs == *this);
}

__device__ __host__ Point Point::operator+(const Point &rhs) const {
  return Point(this->x + rhs.x, this->y + rhs.y);
}

__device__ __host__ void Point::operator+=(const Point &rhs) {
   x += rhs.x;
   y += rhs.y;
}

__device__ __host__ Point Point::operator/(const int div) const {
  return Point(this->x / div, this->y / div);
}

__device__ __host__ void Point::operator/=(const int div) {
  x /= div;
  y /= div;
}

__device__ __host__ Point::Point(const std::string &str) {
  unsigned long coma = str.rfind(',');
  x = std::stof(str.substr(0, coma));
  y = std::stof(str.substr(coma + 1));
}

std::ostream &operator<<(std::ostream &os, const Point &point) {
  os << point.x << ',' << point.y;
  return os;
}

__device__ __host__ float Point::distance_squared_to(Point &other) const {
  float diff_x = x - other.x;
  float diff_y = y - other.y;
  return diff_x * diff_x + diff_y * diff_y;
}
