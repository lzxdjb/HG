#include "head.cuh"
#include <cusparse.h>
#include <cusolverSp.h>

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

void debug(TinyCache *solvergpu)
{
    for (int i = 0; i < horizon; i++)
    {
        // std::cout << "state_vectors = " << solvergpu[i].state1 << std::endl;
        // std::cout << "control_vectors = " << solvergpu[i].control << std::endl;

        // std::cout << "gradient = " << solvergpu[i].gradient << std::endl;
        // std::cout << "Hessian = " << solvergpu[i].Hessian << std::endl;

        // std::cout << "equality = " << solvergpu[i].equality << std::endl;

        // std::cout << "JB1 = \n" << solvergpu[i].JB1 << std::endl;

        // std::cout << "JB2 = \n" << solvergpu[i].JB2 << std::endl;

        // std::cout << "LowerLeftDown1 = \n" << solvergpu[i].LowerLeftDown1 << std::endl;

        // std::cout << "LowerLeftDown2 = \n" << solvergpu[i].LowerLeftDown2 << std::endl;

        // std::cout << "debug = \n" << solvergpu[i].debug << std::endl;

        // std::cout << "shared = \n" << &dshared << std::endl;

        // std::cout << "final_state = " << solvergpu[i].final_state << std::endl;
        // std::cout << "initstate = " << solvergpu[i].initial_state << std::endl;

        // std::cout << "FinalMatrix = \n"
        //           << solvergpu[i].OriginalMatrix << std::endl;
        // std::cout << "FinalColumn = \n"
        //           << solvergpu[i].FinalColumn << std::endl;

        // std::cout << "L = \n" << solvergpu[i].L << std::endl;

        // std::cout << "varible1 = \n" << solvergpu[i].varible1 << std::endl;
        // std::cout << "varible2 = \n"
        //           << solvergpu[i].varible2 << std::endl;

        std::cout << "FirstVarible = \n"<< solvergpu[i].FirstVarible << std::endl;

        // std::cout << "FirstDual = \n" << solvergpu[i].FirstDual << std::endl;

        //  std::cout << "soltuion temp = \n"<< solvergpu[i].solutionTemp << std::endl;

        // std::cout << "convergence = \n"<< solvergpu[i].convergence << std::endl;
    }
}

__global__ void solve_kernel(TinyCache *solver_gpu, double *dshared, double *bigDual)
{

    int idx = threadIdx.x;
    if (idx < horizon)
    {
        __shared__ SharedMatrix shared;

        for (int i = 0; i < horizon * StateShape; i++)
        {
            for (int j = 0; j < horizon; j++)
            {
                shared.row(i)[j] = 0;
            }
        }

        __syncthreads();

        QCost Q = solver_gpu[idx].Q;
        RCost R = solver_gpu[idx].R;

        gradient Allgradient;
        Hessian Hessian;
        equality equality;

        JB JB1;
        JB JB2;

        FinalMatrix finalMatrix;
        FinalColumn finalColumn;

        varible varible1;
        varible varible2;

        FinalMatrix L;
        // solve2();

        double learning_rate = 1;

        tinytype norm;

        JB LowerLeftDown1;
        JB LowerLeftDown2;

        LowerLeftDown1.setZero();
        LowerLeftDown2.setZero();

        // solver_gpu[idx].convergence.topLeftCorner(StateShape, 1) = solver_gpu[idx].state1;
        // solver_gpu[idx].convergence.bottomRightCorner(ControlShape, 1) = solver_gpu[idx].control;

        // while (1)

        for (int i = 0; i < 1; i++)
        {

            Hessian = GetHessian(Q, R);

            if (idx == 0)
            {
                equality = GetEquality(solver_gpu[idx].state1, solver_gpu[idx].control, solver_gpu[idx].initial_state);
            }
            else
            {
                equality = GetEquality(solver_gpu[idx].state1, solver_gpu[idx].control, solver_gpu[idx - 1].state1);
            }

            if (idx != horizon - 1)
            {
                //  printf("asdfasd");

                if (idx == 0)
                {
                    JB1 = GetJB1(solver_gpu[idx].state1, solver_gpu[idx].initial_state);
                    JB2 = GetJB2(solver_gpu[idx].state1, solver_gpu[idx + 1].control);
                }
                else
                {
                    JB1 = GetJB1(solver_gpu[idx].state1, solver_gpu[idx - 1].state1);
                    JB2 = GetJB2(solver_gpu[idx].state1, solver_gpu[idx + 1].control);
                }
            }

            else
            {
                // printf("kkkkkkk");

                JB1 = GetJB1(solver_gpu[idx].state1, solver_gpu[idx - 1].state1);
                JB2.setZero();
            }

            Allgradient = GetGradient(solver_gpu[idx].state1, solver_gpu[idx].control, solver_gpu[idx].final_state, Q, R);

            LowerLeftDown1 = JB1.lazyProduct(PsedoInverse(Hessian));

            LowerLeftDown2 = JB2.lazyProduct(PsedoInverse(Hessian));

            solver_gpu[idx].LowerLeftDown1 = LowerLeftDown1;

            solver_gpu[idx].LowerLeftDown2 = LowerLeftDown2;

            /////////
            temp temp1, temp2, temp3, temp4;

            if (idx != horizon - 1)
            {

                temp1 = -LowerLeftDown1.lazyProduct(JB1.transpose());
                temp2 = -LowerLeftDown1.lazyProduct(JB2.transpose());
                temp3 = -LowerLeftDown2.lazyProduct(JB1.transpose());
                temp4 = -LowerLeftDown2.lazyProduct(JB2.transpose());

                mycopy(&shared, temp1, temp2, temp3, temp4, idx);
            }
            else
            {
                temp1 = -LowerLeftDown1.lazyProduct(JB1.transpose());
                mycopy2(&shared, temp1, idx);
            }

            //////
            solver_gpu[idx].FirstVarible = - Allgradient;

            __syncthreads();

            state SolutionTemp;
            if (idx == 0)
            {
                SolutionTemp = LowerLeftDown1.lazyProduct(solver_gpu[idx].FirstVarible);
            }
            else
            {
                // printf("asdfasdfas");
                SolutionTemp = LowerLeftDown1.lazyProduct(solver_gpu[idx].FirstVarible);

                SolutionTemp += solver_gpu[idx - 1].LowerLeftDown2.lazyProduct(solver_gpu[idx - 1].FirstVarible);

                // printf("olver_gpu[idx].FirstVarible = %f\n" , LowerLeftDown1.row(0)[0]);
                // printf("solver_gpu[idx].FirstVarible = %f \n" , solver_gpu[idx].FirstVarible.row(0)[0]);
            }

            solver_gpu[idx].solutionTemp = SolutionTemp;

            solver_gpu[idx].FirstDual = -equality - SolutionTemp;
        }

        __syncthreads();

        solver_gpu[idx].equality = equality;
        solver_gpu[idx].JB1 = JB1;
        solver_gpu[idx].JB2 = JB2;
        solver_gpu[idx].Hessian = Hessian;
        solver_gpu[idx].gradient = Allgradient;
        // solver_gpu[idx].FinalMatrix = finalMatrix;
        // solver_gpu[idx].FinalColumn = finalColumn;

        // solver_gpu[idx].L = L;
        // solver_gpu[idx].varible1 = varible1;
        // solver_gpu[idx].varible2 = varible2;

        for (int i = 0; i < horizon * StateShape; i++)
        {
            for (int j = 0; j < horizon * StateShape; j++)
            {
                dshared[i * horizon * StateShape + j] = shared.row(i)[j];
            }
        }

        for (int i = 0; i < StateShape; i++)
        {
            bigDual[idx * StateShape + i] = solver_gpu[idx].FirstDual[i];
        }
    }
}

__global__ void Second_solve_kernel(TinyCache *solver_gpu , double * d_x)
{
    int idx = threadIdx.x;
    if (idx < horizon)
    {
       SecondPhaseCopy(&solver_gpu[idx].FirstDual  , d_x , idx);

       convergence SolutionTemp;

            if (idx == horizon - 1)
            {
                SolutionTemp =  solver_gpu[idx].JB1.transpose().lazyProduct(solver_gpu[idx].FirstDual);
            }
            else
            {
                // printf("asdfasdfas");
                SolutionTemp = solver_gpu[idx].JB1.transpose().lazyProduct(solver_gpu[idx].FirstDual);

                SolutionTemp += solver_gpu[idx].JB2.transpose().lazyProduct(solver_gpu[idx + 1].FirstDual);

            }
            solver_gpu[idx].FirstVarible -= SolutionTemp;
            solver_gpu[idx].FirstVarible = PsedoInverse(solver_gpu[idx].Hessian).lazyProduct(solver_gpu[idx].FirstVarible);


            // solver_gpu[idx].convergence = SolutionTemp;


       


    }
}


void tiny_solve_cuda(TinyCache *cache, tinytype *shared, tinytype *bigDual)
{
    TinyCache *solver_gpu;

    double *dshared;

    double *d_dual; // FirstStageDual
    cudaMalloc(&d_dual, StateShape * horizon * sizeof(double));

    ////

    checkCudaErrors(cudaMalloc((void **)&solver_gpu, sizeof(TinyCache) * horizon));
    checkCudaErrors(cudaMemcpy(solver_gpu, cache, sizeof(TinyCache) * horizon, cudaMemcpyHostToDevice));

    /////
    checkCudaErrors(cudaMalloc((void **)&dshared, sizeof(SharedMatrix)));
    checkCudaErrors(cudaMemcpy(dshared, shared, sizeof(SharedMatrix),
                               cudaMemcpyHostToDevice));

    cusparseHandle_t handle;
    (cusparseCreate(&handle));
    cusparseMatDescr_t descrA;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    int Nrows = StateShape * horizon;
    int Ncols = StateShape * horizon;
    int N = Nrows;

    int nnz = 0;
    const int lda = Nrows;
    double *d_A;
    int *d_A_RowIndices;
    int *d_A_ColIndices;

    int *d_nnzPerVector;
    checkCudaErrors(cudaMalloc(&d_nnzPerVector, Nrows * sizeof(*d_nnzPerVector)));
    cusolverSpHandle_t solver_handle;
    cusolverSpCreate(&solver_handle);
    int singularity;
    double *d_x;
    (cudaMalloc(&d_x, Ncols * sizeof(double)));

    for (int i = 0; i < 1; i++)
    {

        solve_kernel<<<1, horizon>>>(solver_gpu, dshared, d_dual);
        checkCudaErrors(cudaDeviceSynchronize());

        //////

        cusparseDnnz(handle, CUSPARSE_DIRECTION_ROW, Nrows, Ncols, descrA, dshared, lda, d_nnzPerVector, &nnz);

        (cudaMalloc(&d_A, nnz * sizeof(*d_A)));
        (cudaMalloc(&d_A_RowIndices, (Nrows + 1) * sizeof(*d_A_RowIndices)));
        (cudaMalloc(&d_A_ColIndices, nnz * sizeof(*d_A_ColIndices)));

        (cusparseDdense2csr(handle, Nrows, Ncols, descrA, dshared, lda, d_nnzPerVector, d_A, d_A_RowIndices, d_A_ColIndices));

        (cusolverSpDcsrlsvqr(solver_handle, N, nnz, descrA, d_A, d_A_RowIndices, d_A_ColIndices, d_dual, 0.000001, 0, d_x, &singularity));

    checkCudaErrors(cudaDeviceSynchronize());

        Second_solve_kernel<<<1, horizon>>>(solver_gpu , d_x);
        checkCudaErrors(cudaDeviceSynchronize());

        /////
    }

    checkCudaErrors(cudaMemcpy(cache, solver_gpu, sizeof(TinyCache) * horizon, cudaMemcpyDeviceToHost));

    /////
    debug(cache);


    checkCudaErrors(cudaMemcpy(shared, dshared, sizeof(SharedMatrix), cudaMemcpyDeviceToHost));

    /////

    checkCudaErrors(cudaMemcpy(bigDual, d_dual, sizeof(TempBigDual), cudaMemcpyDeviceToHost));


}
