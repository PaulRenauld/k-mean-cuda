//
// Created by Paul Renauld on 2019-04-17.
//

#ifndef CPP_POINT_H
#define CPP_POINT_H


#include <string>
#include <ostream>

class Point {
  public:
    Point(float x, float y) : x(x), y(y) {
    }

    Point() : x(0), y(0) {
    }

    Point(const std::string &str);

    bool operator==(const Point &rhs) const;

    bool operator!=(const Point &rhs) const;

    Point operator+(const Point &rhs) const;

    Point operator/(const int div) const;

    friend std::ostream &operator<<(std::ostream &os, const Point &point);

  private:
    float x, y;

};


#endif //CPP_POINT_H
