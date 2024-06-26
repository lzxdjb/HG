#include "head.cuh"

void debug(TinyCache *solvergpu)
{
    for (int i = 0; i < horizon + 1; i++)
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
        // std::cout<<"cache = " <<solvergpu[i].cache<<std::endl;

        // std::cout<<"SparseCache = "<<solvergpu[i].SparseCache<<std::endl;

        // std::cout<<"h_A_RowIndices = " <<solvergpu[i].h_A_RowIndices<<std::endl;
        //   std::cout<<"h_A_ColIndices = " <<solvergpu[i].h_A_ColIndices<<std::endl;
    }
}

__global__ void solve_kernel(TinyCache *solver_gpu, double *dshared, double *bigDual, QCost Q, RCost R, state init_state, state final_state)
{

    int idx = threadIdx.x;
    if (idx < horizon + 1 && idx > 0)
    {
        __shared__ SharedMatrix shared;

        empty(&shared);

        __syncthreads();

        gradient Allgradient;
        Hessian Hessian;
        equality equality;

        JB JB1;
        JB JB2;

        // varible varible1;
        // varible varible2;

        JB LowerLeftDown1;
        JB LowerLeftDown2;

        LowerLeftDown1.setZero();
        LowerLeftDown2.setZero();

        Hessian = GetHessian(Q, R);

    
        equality = GetEquality(solver_gpu[idx].state1, solver_gpu[idx].control, solver_gpu[idx - 1].state1);
        
        // printf("idx = %d " , idx);

        if (idx != horizon )
        {

            JB1 = GetJB1(solver_gpu[idx].state1, solver_gpu[idx - 1].state1);
            JB2 = GetJB2(solver_gpu[idx].state1, solver_gpu[idx].control);
            
        }

        else
        {
            // printf("asdfasdf");
            JB1 = GetJB1(solver_gpu[idx].state1, solver_gpu[idx - 1].state1);
            JB2.setZero();
        }

        Allgradient = GetGradient(solver_gpu[idx].state1, solver_gpu[idx].control, final_state, Q, R);

        LowerLeftDown1 = JB1.lazyProduct(PsedoInverse(Hessian));
        LowerLeftDown2 = JB2.lazyProduct(PsedoInverse(Hessian));
        
        solver_gpu[idx].LowerLeftDown2 = LowerLeftDown2;

        temp temp1, temp2, temp3, temp4;

        if (idx != horizon)
        {

            temp1 = -LowerLeftDown1.lazyProduct(JB1.transpose());
            temp2 = -LowerLeftDown1.lazyProduct(JB2.transpose());
            temp3 = -LowerLeftDown2.lazyProduct(JB1.transpose());
            temp4 = -LowerLeftDown2.lazyProduct(JB2.transpose());
            mycopy(&shared, temp1, temp2, temp3, temp4, idx);

            solver_gpu[idx].cache.topLeftCorner(StateShape , StateShape) += -LowerLeftDown1.lazyProduct(JB1.transpose());

            solver_gpu[idx].cache.topRightCorner(StateShape , StateShape) += -LowerLeftDown1.lazyProduct(JB2.transpose());

            solver_gpu[idx + 1].cache.topLeftCorner(StateShape , StateShape) += -LowerLeftDown2.lazyProduct(JB1.transpose());

            solver_gpu[idx + 1].cache.topRightCorner(StateShape , StateShape) += -LowerLeftDown2.lazyProduct(JB2.transpose());
        }
        else
        {
            temp1 = -LowerLeftDown1.lazyProduct(JB1.transpose());
            mycopy2(&shared, temp1 , idx);

            solver_gpu[idx].cache.topRightCorner(StateShape , StateShape) += -LowerLeftDown1.lazyProduct(JB1.transpose());
        }

        // dense_to_csr(&solver_gpu[idx].cache , &solver_gpu[idx].SparseCache , StateShape , StateShape * 2 , &solver_gpu[idx].h_A_RowIndices , &solver_gpu[idx].h_A_ColIndices);



        //////
        solver_gpu[idx].FirstVarible = -Allgradient;

        __syncthreads();

        state SolutionTemp;
        if (idx == 1)
        {
            SolutionTemp = LowerLeftDown1.lazyProduct(solver_gpu[idx].FirstVarible);
        }
        else
        {
            // printf("asdfasdfas");
            SolutionTemp = LowerLeftDown1.lazyProduct(solver_gpu[idx].FirstVarible);

            SolutionTemp += solver_gpu[idx - 1].LowerLeftDown2.lazyProduct(solver_gpu[idx - 1].FirstVarible);
        }

        solver_gpu[idx].FirstDual = -equality - SolutionTemp;
        solver_gpu[idx].JB1 = JB1;
        solver_gpu[idx].JB2 = JB2;
        solver_gpu[idx].Hessian = Hessian;

        for (int i = 0; i < horizon * StateShape; i++)
        {
            for (int j = 0; j < horizon * StateShape; j++)
            {
                dshared[i * horizon * StateShape + j] = shared.row(i)[j];
            }
        }

        for (int i = 0; i < StateShape; i++)
        {
            bigDual[(idx - 1) * StateShape + i] = solver_gpu[idx].FirstDual[i];
        }
    }
}

__global__ void Second_solve_kernel(TinyCache *solver_gpu, double *d_x)
{
    int idx = threadIdx.x;
    if (idx < horizon + 1 && idx >0)
    {
        SecondPhaseCopy(&solver_gpu[idx].FirstDual, d_x, idx);

        convergence SolutionTemp;

        if (idx == horizon)
        {
            SolutionTemp = solver_gpu[idx].JB1.transpose().lazyProduct(solver_gpu[idx].FirstDual);
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

void tiny_solve_cuda(TinyCache *cache, tinytype *shared, tinytype *bigDual, QCost Q, RCost R, state init_state, state final_state)
{
    TinyCache *solver_gpu;

    double *dshared;

    double *d_dual; // FirstStageDual
    cudaMalloc(&d_dual, StateShape * horizon * sizeof(double));

    ////

    checkCudaErrors(cudaMalloc((void **)&solver_gpu, sizeof(TinyCache) * (horizon + 1) ));
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

        solve_kernel<<<1, horizon + 1>>>(solver_gpu, dshared, d_dual, Q, R, init_state, final_state);
        checkCudaErrors(cudaDeviceSynchronize());

        cusparseDnnz(handle, CUSPARSE_DIRECTION_ROW, Nrows, Ncols, descrA, dshared, lda, d_nnzPerVector, &nnz);

        (cudaMalloc(&d_A, nnz * sizeof(*d_A)));
        (cudaMalloc(&d_A_RowIndices, (Nrows + 1) * sizeof(*d_A_RowIndices)));
        (cudaMalloc(&d_A_ColIndices, nnz * sizeof(*d_A_ColIndices)));

        (cusparseDdense2csr(handle, Nrows, Ncols, descrA, dshared, lda, d_nnzPerVector, d_A, d_A_RowIndices, d_A_ColIndices));

        (cusolverSpDcsrlsvqr(solver_handle, N, nnz, descrA, d_A, d_A_RowIndices, d_A_ColIndices, d_dual, 0.000001, 0, d_x, &singularity));

        //////////

        /////
        double *h_A = (double *)malloc(nnz * sizeof(*h_A));
        int *h_A_RowIndices = (int *)malloc((Nrows + 1) * sizeof(*h_A_RowIndices));
        int *h_A_ColIndices = (int *)malloc(nnz * sizeof(*h_A_ColIndices));
        (cudaMemcpy(h_A, d_A, nnz * sizeof(*h_A), cudaMemcpyDeviceToHost));
        /////
        (cudaMemcpy(h_A_RowIndices, d_A_RowIndices, (Nrows + 1) * sizeof(*h_A_RowIndices), cudaMemcpyDeviceToHost));
        (cudaMemcpy(h_A_ColIndices, d_A_ColIndices, nnz * sizeof(*h_A_ColIndices), cudaMemcpyDeviceToHost));

        // std::cout << "realAnswer : " << std::endl;

        // for (int i = 0; i < nnz; ++i)
        //     printf("A[%i] = %.0f ", i, h_A[i]);
        // printf("\n");

        // for (int i = 0; i < (Nrows + 1); ++i)
        //     printf("h_A_RowIndices[%i] = %i \n", i, h_A_RowIndices[i]);
        // printf("\n");

        // for (int i = 0; i < nnz; ++i)
        //     printf("h_A_ColIndices[%i] = %i \n", i, h_A_ColIndices[i]);

        //////////

        Second_solve_kernel<<<1, horizon + 1>>>(solver_gpu, d_x);
        checkCudaErrors(cudaDeviceSynchronize());
    }

    checkCudaErrors(cudaMemcpy(cache, solver_gpu, sizeof(TinyCache) * (horizon + 1), cudaMemcpyDeviceToHost));
    debug(cache);

    checkCudaErrors(cudaMemcpy(shared, dshared, sizeof(SharedMatrix), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(bigDual, d_dual, sizeof(TempBigDual), cudaMemcpyDeviceToHost));
}


