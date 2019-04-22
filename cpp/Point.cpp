//
// Created by Paul Renauld on 2019-04-17.
//

#include "Point.h"

bool Point::operator==(const Point &rhs) const {
  return x == rhs.x &&
         y == rhs.y;
}

bool Point::operator!=(const Point &rhs) const {
  return !(rhs == *this);
}

Point Point::operator+(const Point &rhs) const {
  return Point(this->x + rhs.x, this->y + rhs.y);
}

Point::Point(const std::string &str) {
  unsigned long coma = str.rfind(',');
  x = stof(str.substr(0, coma));
  y = stof(str.substr(coma + 1));
}

std::ostream &operator<<(std::ostream &os, const Point &point) {
  os << point.x << ',' << point.y;
  return os;
}
