//
// Created by Paul Renauld on 2019-04-17.
//

#ifndef CPP_POINT_H
#define CPP_POINT_H


#include <string>
#include <ostream>

#if CUDA==1
    #include <cuda.h>
    #include <cuda_runtime.h>
#else
    #define __device__
    #define __host__
#endif

class Point {
  public:
    __device__ __host__ Point(float x, float y) : x(x), y(y) {}

    __device__ __host__ Point() : x(0), y(0) {}

    __device__ __host__ Point(const std::string &str);

    __device__ __host__ bool operator==(const Point &rhs) const;

    __device__ __host__ bool operator!=(const Point &rhs) const;

    __device__ __host__ Point operator+(const Point &rhs) const;

    __device__ __host__ void operator+=(const Point &rhs);

    __device__ __host__ Point operator/(int div) const;

    __device__ __host__ void operator/=(int div);

    __device__ __host__ float distance_squared_to(Point &other) const;

    friend std::ostream &operator<<(std::ostream &os, const Point &point);

  private:
    float x, y;

};


#endif //CPP_POINT_H
