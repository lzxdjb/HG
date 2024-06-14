#pragma once

// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "type.cuh"

using std::vector;
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

__global__ void solve_kernel(TinyCache *solver_gpu);
void tiny_solve_cuda(TinyCache *cache);

__device__ static inline double dot_product(double *a, double *b, int number)
{
    double result = 0;
    for (int i = 0; i < number; i++)
    {
        result += a[i] * b[i];
    }
    return result;
}

__device__ static inline void copy(double *a, gradient b, int number)
{
    double result = 0;
    for (int i = 0; i < number; i++)
    {
        a[i] = b[i];
    }
}

__device__ static inline gradient GetGradient(state state1, control control, final_state final_state, QCost Q, RCost R)
{

    Eigen::VectorXd Q_result;

    Eigen::VectorXd tempj = (state1 - final_state);

    Q_result = 2 * Q.lazyProduct(tempj);

    Eigen::VectorXd R_result;
    R_result = 2 * R.lazyProduct(control);

    Matrix<double, ControlShape + StateShape, 1> temp;

    temp.topLeftCorner(StateShape, 1) = Q_result;
    temp.bottomRightCorner(ControlShape, 1) = R_result;

    return temp ;
}

__device__ static inline Hessian GetHessian(QCost Q, RCost R)
{
    int totalSize = StateShape + ControlShape;
    MatrixXd M = MatrixXd::Zero(totalSize, totalSize);

    // Place Q in the top-left corner
    M.topLeftCorner(StateShape, StateShape) = Q;

    // Place R in the bottom-right corner
    M.bottomRightCorner(ControlShape, ControlShape) = R;
    return M * 2;
}

__device__ static inline equality GetEquality(state state1, control control, state initial)
{
    equality temp;
    temp[0] = (state1[0] - cos(initial[2]) * control[0]) * T;
    temp[1] = (state1[1] - sin(initial[2]) * control[0]) * T;
    temp[2] = (state1[2] - control[1]) * T;

    return temp;
}

__device__ static inline JB GetJB(state state1, control control, state initial)
{
    StateJB StateJB;
    StateJB << 1, 0, 0,
        0, 1, 0,
        0, 0, 1;
    ControlJB ControlJB;
    ControlJB << -cos(initial[2]), 0,
        -sin(initial[2]), 0,
        0, -1;
    JB JB;
    JB.topLeftCorner(StateShape, StateShape) = StateJB;
    JB.bottomRightCorner(StateShape, ControlShape) = ControlJB;

    return JB;
}

__device__ static inline FinalMatrix GetFinalMatrix(Hessian Hessian, JB JB)
{
    FinalMatrix FinalMatrix;

    FinalMatrix.setZero();

    // Place H in the top-left corner
    FinalMatrix.topLeftCorner(Hessian.rows(), Hessian.cols()) = Hessian;

    // Place J^T in the top-right corner
    FinalMatrix.topRightCorner(JB.cols(), JB.rows()) = JB.transpose();

    // Place J in the bottom-left corner
    FinalMatrix.bottomLeftCorner(JB.rows(), JB.cols()) = JB;

    return FinalMatrix;
}

__device__ static inline FinalColumn GetFinalColumn(gradient Allgradient, equality equality)
{
    FinalColumn FinalColumn;
    FinalColumn.topLeftCorner(Allgradient.rows(), 1) = Allgradient;
    FinalColumn.bottomLeftCorner(equality.rows(), 1) = equality;

    return - FinalColumn;
}

__device__ static inline void RowElimination(FinalMatrix *finalmatrix, FinalMatrix *L, int startrow, int startcolumn, int desrow, int descolumn)
{
    double index = 0;

    index = 1 / finalmatrix->row(startrow)[startcolumn] * finalmatrix->row(desrow)[descolumn];

    finalmatrix->row(desrow) -= finalmatrix->row(startrow) * index;

    L->row(desrow)[descolumn] = L->row(startrow)[startcolumn] * index;
}

__device__ static inline void solve1(FinalMatrix *L, FinalMatrix *finalMatrix, FinalColumn *varible1, FinalColumn *varible2, FinalColumn *finalColumn)
{

    for (int i = 0; i < finalColumn->size(); i++)
    {
        varible1->row(i)[0] = finalColumn->row(i)[0];
    }

    for (int i = 0; i < varible1->size(); i++)
    {

        for (int j = 0; j < i; j++)
        {
            varible1->row(i)[0] -= L->row(i)[j] * varible1->row(j)[0];
        }
        varible1->row(i)[0] /= L->row(i)[i];
    }

    for (int i = 0; i < finalColumn->size(); i++)
    {
        varible2->row(i)[0] = varible1->row(i)[0];
    }
    for (int i = varible1->size() - 1; i >= 0; i--)
    {

        for (int j = varible1->size() - 1; j > i; j--)
        {
            varible2->row(i)[0] -= finalMatrix->row(i)[j] * varible2->row(j)[0];
        }
        varible2->row(i)[0] /= finalMatrix->row(i)[i];
    }


    // Eigen::SparseMatrix<double> sparseMat = finalMatrix->sparseView();

}


__device__ static inline void ChangefinalMatrix(FinalMatrix *finalmatrix , state * initial_state)
{

}