#include "DeviceFunction.cuh"

__device__ void copy(double *a, gradient b, int number)
{
    double result = 0;
    for (int i = 0; i < number; i++)
    {
        a[i] = b[i];
    }
}

__device__ gradient GetGradient(state state1, control control, final_state final_state, QCost Q, RCost R)
{

    Eigen::VectorXd Q_result;

    Eigen::VectorXd tempj = (state1 - final_state);

    Q_result = 2 * Q.lazyProduct(tempj);

    Eigen::VectorXd R_result;
    R_result = 2 * R.lazyProduct(control);

    Matrix<double, ControlShape + StateShape, 1> temp;

    temp.topLeftCorner(StateShape, 1) = Q_result;
    temp.bottomRightCorner(ControlShape, 1) = R_result;

    return temp;
}

__device__ Hessian GetHessian(QCost Q, RCost R)
{
    int totalSize = StateShape + ControlShape;
    MatrixXd M = MatrixXd::Zero(totalSize, totalSize);

    // Place Q in the top-left corner
    M.topLeftCorner(StateShape, StateShape) = Q;

    // Place R in the bottom-right corner
    M.bottomRightCorner(ControlShape, ControlShape) = R;
    return M * 2;
}

__device__ equality GetEquality(state state1, control control, state initial)
{
    equality temp;
    temp[0] = (state1[0] - initial[0] - cos(initial[2]) * control[0]) * T;
    temp[1] = (state1[1] - initial[1] - sin(initial[2]) * control[0]) * T;
    temp[2] = (state1[2] - initial[2] - control[1]) * T;

    return temp;
}

__device__ JB GetJB1(state state1, state initial)
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

__device__ JB GetJB2(state state1, control control)
{
    StateJB StateJB;
    StateJB << -1, 0, control[0] * sin(state1[2]),
        0, -1, - control[0] * cos(state1[2]),
        0, 0, -1;
    ControlJB ControlJB;
    ControlJB << 0, 0,
        0, 0,
        0, 0;
    JB JB;
    JB.topLeftCorner(StateShape, StateShape) = StateJB;
    JB.bottomRightCorner(StateShape, ControlShape) = ControlJB;

    return JB;
}

__device__ Hessian PsedoInverse(Hessian hessian)
{
    Hessian temp;
    temp.setZero();
    for (int i = 0; i < StateShape + ControlShape; i++)
    {
        temp.row(i)[i] = 1 / hessian.row(i)[i];
    }
    return temp;
}

__device__ void mycopy(SharedMatrix *shared, temp temp1, temp temp2 , temp temp3 , temp temp4 ,  int idx)
{
    int base = idx * StateShape;

    for (int i = 0; i < StateShape; i++)
    {
        for (int j = 0; j < StateShape; j++)
        {
            MyatomicAdd(&shared->row(i + base)[j + base], temp1.row(i)[j]);

       
            MyatomicAdd(&shared->row(i + base )[j + base + StateShape], temp2.row(i)[j]);

            MyatomicAdd(&shared->row(i + base + StateShape)[j + base], temp3.row(i)[j]);

            MyatomicAdd(&shared->row(i + base + StateShape)[j + base + StateShape], temp4.row(i)[j]);
            
        }
    }
}

__device__ void mycopy2(SharedMatrix *shared, temp temp1, int idx)
{
    int base = idx * StateShape;

    for (int i = 0; i < StateShape; i++)
    {
        for (int j = 0; j < StateShape; j++)
        {
            MyatomicAdd(&shared->row(i + base)[j + base], temp1.row(i)[j]);        
        }
    }
}


__device__ void DebugCopy(SharedMatrix *shared, SharedMatrix *debug)
{

    for (int i = 0; i <horizon *  StateShape; i++)
    {
        for (int j = 0; j < horizon * StateShape; j++)
        {
           debug->row(i)[j] = shared->row(i)[j];
        }
    }
}


__device__ void SecondPhaseCopy(FirstPhaseDual * FirstDual , double * d_x ,int idx) 
{
    for(int i = 0 ; i < StateShape ; i ++)
    {
        FirstDual->row(i)[0] = d_x[i + idx * StateShape];
    }

}



__device__ double MyatomicAdd(double *address, double val)
{
    unsigned long long int *address_as_ull =
        (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull, assumed;
    do
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                                             __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__device__ double empty(SharedMatrix *matrix)
{
    for(int i = 0 ; i < StateShape * horizon ; i ++)
    {
        for (int j = 0 ; j < StateShape * horizon ; j ++)
        {
            matrix->row(i)[j] = 0;
        }
    }

}
