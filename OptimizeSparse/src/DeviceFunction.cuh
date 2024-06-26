#pragma once

#include <iostream>
#include <Eigen/Dense>
#include "type.cuh"

using namespace Eigen;

#define checkCudaErrors(x) check((x), #x, __FILE__, __LINE__)

template <typename T>
void check(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
                static_cast<unsigned int>(result), cudaGetErrorName(result), func);
    }
}




__device__ double MyatomicAdd(double* address, double val);


__device__ void copy(double *a, gradient b, int number);


__device__ gradient GetGradient(state state1, control control, final_state final_state, QCost Q, RCost R);

__device__ Hessian GetHessian(QCost Q, RCost R);



__device__ equality GetEquality(state state1, control control, state initial);

__device__ JB GetJB1(state state1, state initial);

__device__ JB GetJB2(state state1, control control);


__device__ Hessian PsedoInverse(Hessian hessian);

__device__ void mycopy(SharedMatrix *shared, temp temp1, temp temp2 , temp temp3 , temp temp4 ,  int idx);

__device__ void mycopy2(SharedMatrix *shared, temp temp1, int idx);

__device__ void DebugCopy(SharedMatrix *shared, SharedMatrix *debug);

__device__ void SecondPhaseCopy(FirstPhaseDual * FirstDual , double * d_x ,int idx) ;


__device__ double MyatomicAdd(double *address, double val);


__device__ double empty(SharedMatrix *matrix);