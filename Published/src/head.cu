#include "head.cuh"
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
void debug(TinyCache *solvergpu)
{
    for (int i = 1; i < horizon + 1; i++)
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

        // std::cout << "FirstVarible = \n"<< solvergpu[i].FirstVarible << std::endl;

        // std::cout << "FirstDual = \n" << solvergpu[i].FirstDual << std::endl;

        //  std::cout << "soltuion temp = \n"<< solvergpu[i].solutionTemp << std::endl;

        // std::cout << "convergence = \n"<< solvergpu[i].convergence << std::endl;
        // std::cout<<"cache1 \n= " <<solvergpu[i].cache1<<std::endl;
        // std::cout<<"cache2 \n= " <<solvergpu[i].cache2<<std::endl;
        // std::cout<<"cache3 \n= " <<solvergpu[i].cache3<<std::endl;
        // std::cout<<"nnz = "<<solvergpu[i].nnz<<std::endl;
        // std::cout<<"SparseCache = "<<solvergpu[i].SparseCache.transpose()<<std::endl;
        // std::cout<<"h_A_RowIndices = " <<solvergpu[i].h_A_RowIndices.transpose()<<std::endl;
        // std::cout<<"h_A_ColIndices = " <<solvergpu[i].h_A_ColIndices.transpose()<<std::endl;

        //  std::cout<<"\n";
    }
}

__global__ void sparse_represent(TinyCache *solver_gpu, int *index ,  double * sparseMatix , int * RowIndices , int * ColIndices)
{
    int idx = threadIdx.x;
    if(idx < horizon + 1 && idx > 0)
    {
        SparsecopyToPointer(solver_gpu[idx].SparseCache , sparseMatix , 0  ,index[idx] - index[idx - 1] ,  index[idx - 1]);
        

        if(idx == 1)
        {
            RowIndicescopyToPointer(solver_gpu[idx].h_A_RowIndices , RowIndices , 0 , StateShape + 1 , (idx- 1) * StateShape);
        }
        else{

            // printf("idx = %d" , idx);
            // printf("index[idx = %d] = %d" ,idx, index[idx]);
            // printf("index[idx - 1 = %d] = %d" , idx - 1, index[idx - 1]);

            for(int i = 1 ; i < StateShape + 1 ; i ++)
            {
                solver_gpu[idx].h_A_RowIndices.row(i)[0] += index[idx - 1];
            }
             RowIndicescopyToPointer(solver_gpu[idx].h_A_RowIndices , RowIndices , 1 , StateShape , (idx- 1) * StateShape + 1);
        }

        if(idx < 3)
        {
            // printf("asdfasdfas");
            ColIndicescopyToPointer(solver_gpu[idx].h_A_ColIndices , ColIndices , 0 , index[idx] - index[idx - 1] , index[idx - 1]);
        }
        else{
            // printf("kkkkkkkkk");

            for(int i = 0 ; i < index[idx] - index[idx - 1] ; i ++)
            {
                solver_gpu[idx].h_A_ColIndices.row(i)[0] += StateShape * (idx - 2);
            }

            ColIndicescopyToPointer(solver_gpu[idx].h_A_ColIndices , ColIndices , 0 , index[idx] - index[idx - 1] , index[idx - 1]);
        }

    }
}

__global__ void solve_kernel(TinyCache *solver_gpu, double *dshared, double *bigDual, QCost Q, RCost R, state init_state, state final_state , int* d_index)
{

    int idx = threadIdx.x;
    if (idx < horizon + 1 && idx > 0)
    {
        __shared__ SharedMatrix shared;

        empty(&shared);

        // __syncthreads();

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

        if (idx == 1 && horizon != 1)
        {

            temp1 = -LowerLeftDown1.lazyProduct(JB1.transpose());
            temp2 = -LowerLeftDown1.lazyProduct(JB2.transpose());
            temp3 = -LowerLeftDown2.lazyProduct(JB1.transpose());
            temp4 = -LowerLeftDown2.lazyProduct(JB2.transpose());
            mycopy(&shared, temp1, temp2, temp3, temp4, idx);
            // printf("idx = %d " , idx);
            solver_gpu[idx].cache1.topLeftCorner(StateShape , StateShape) += -LowerLeftDown1.lazyProduct(JB1.transpose());

            solver_gpu[idx].cache2.topLeftCorner(StateShape , StateShape) += -LowerLeftDown1.lazyProduct(JB2.transpose());

            solver_gpu[idx + 1].cache1.topLeftCorner(StateShape , StateShape) += -LowerLeftDown2.lazyProduct(JB1.transpose());

            solver_gpu[idx + 1].cache2.topLeftCorner(StateShape , StateShape) += -LowerLeftDown2.lazyProduct(JB2.transpose());
        }

        else if (idx < horizon && idx > 1)
        {

            temp1 = -LowerLeftDown1.lazyProduct(JB1.transpose());
            temp2 = -LowerLeftDown1.lazyProduct(JB2.transpose());
            temp3 = -LowerLeftDown2.lazyProduct(JB1.transpose());
            temp4 = -LowerLeftDown2.lazyProduct(JB2.transpose());
            mycopy(&shared, temp1, temp2, temp3, temp4, idx);

            solver_gpu[idx].cache2.topLeftCorner(StateShape , StateShape) += -LowerLeftDown1.lazyProduct(JB1.transpose());

            // printf("idx = %d " , idx);

            solver_gpu[idx].cache3.topRightCorner(StateShape , StateShape) += -LowerLeftDown1.lazyProduct(JB2.transpose());

            solver_gpu[idx + 1].cache1.topLeftCorner(StateShape , StateShape) += -LowerLeftDown2.lazyProduct(JB1.transpose());

            solver_gpu[idx + 1].cache2.topRightCorner(StateShape , StateShape) += -LowerLeftDown2.lazyProduct(JB2.transpose());

        }
        else
        {
            temp1 = -LowerLeftDown1.lazyProduct(JB1.transpose());
            mycopy2(&shared, temp1 , idx);

            solver_gpu[idx].cache2.topRightCorner(StateShape , StateShape) += -LowerLeftDown1.lazyProduct(JB1.transpose());
          
    
        }

        //////



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

        solver_gpu[idx].nnz = dense_to_csr(solver_gpu[idx].cache1 , solver_gpu[idx].cache2 , solver_gpu[idx].cache3 ,&solver_gpu[idx].SparseCache , StateShape , StateShape * 3 , &solver_gpu[idx].h_A_RowIndices , &solver_gpu[idx].h_A_ColIndices);

        __syncthreads();
        d_index[idx] = solver_gpu[idx].nnz;
        
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
    if (idx < horizon + 1 && idx > 0)
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
    // printf("addr %d\n", solver_gpu);
    checkCudaErrors(cudaMemcpy(solver_gpu, cache, sizeof(TinyCache) * (horizon + 1), cudaMemcpyHostToDevice));

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

   

////////##########
    // DebugSparse h_debug_sparse;
    // DebugSparse *d_debug_sparse;
    // h_debug_sparse.setZero();
    // checkCudaErrors(cudaMalloc((void**)&d_debug_sparse, sizeof(DebugSparse)));
    // cudaMemcpy(d_debug_sparse, &h_debug_sparse, sizeof(DebugSparse), cudaMemcpyHostToDevice);

////////########## Do not delete

    int h_index[horizon + 1] = {0};
    int *d_index;
    checkCudaErrors(cudaMalloc((void**)&d_index, sizeof(int) * (horizon + 1)));
    cudaMemcpy(d_index, &h_index, sizeof(int) * (horizon + 1) , cudaMemcpyHostToDevice);

    double *d_sparsematrix;
    int *d_RowIndices;
    int *d_ColIndices;
    checkCudaErrors(cudaMalloc((void**)&d_sparsematrix, sizeof(double) * (StateShape * StateShape * horizon * 3)));
    checkCudaErrors(cudaMalloc((void**)&d_RowIndices, sizeof(int) *  (StateShape * horizon + 1)));
    checkCudaErrors(cudaMalloc((void**)&d_ColIndices, sizeof(int) * (StateShape * StateShape * horizon * 3 )));
    

//@@@@@@@@@@@

    for (int i = 0; i < 1; i++)
    {

        solve_kernel<<<1, horizon + 1>>>(solver_gpu, dshared, d_dual, Q, R, init_state, final_state , d_index);
        checkCudaErrors(cudaDeviceSynchronize());

        thrust::inclusive_scan(thrust::device_pointer_cast(d_index), thrust::device_pointer_cast(d_index) + horizon + 1, thrust::device_pointer_cast(d_index)); // in-place scan
        


////###########$$$$$$$$$$$$ TEST
       

        std::cout << "My answer : " << std::endl;

        sparse_represent<<<1 , horizon + 1>>>(solver_gpu , d_index , d_sparsematrix , d_RowIndices , d_ColIndices);

    double *h_sparsematrix = (double *)malloc((StateShape * StateShape * horizon * 3) * sizeof(double));;
    int *h_RowIndices = (int * )malloc(sizeof(int) * (StateShape * horizon +1));
    int *h_ColIndices =(int * )malloc(sizeof(int) * (StateShape * StateShape * horizon * 3 ));

    (cudaMemcpy(h_sparsematrix, d_sparsematrix, (StateShape * StateShape * horizon  * 3) * sizeof(double), cudaMemcpyDeviceToHost));
        (cudaMemcpy(h_RowIndices, d_RowIndices, sizeof(int) * (StateShape * horizon + 1) , cudaMemcpyDeviceToHost));
        (cudaMemcpy(h_ColIndices, d_ColIndices, (sizeof(int) * (StateShape * StateShape * horizon * 3 )), cudaMemcpyDeviceToHost));


        checkCudaErrors(cudaMemcpy(&h_index, d_index, sizeof(int) * (horizon + 1), cudaMemcpyDeviceToHost));
        // for(int i = 0 ; i < horizon + 1 ; i ++)
        // {
        //     std::cout<<h_index[i]<<" ";
        // }
        // std::cout<<std::endl;

        // for (int i = 0; i < 41; ++i)
        //     printf("d_sparsematrix[%i] = %f ", i, h_sparsematrix[i]);
        // printf("\n");
        printf("h_A_RowIndices] = \n ");
        for (int i = 0; i < StateShape * horizon + 1; ++i)
            printf("%i ", h_RowIndices[i]);
        printf("\n");
        printf("h_ColIndices = \n ");
        for (int i = 0; i < (41); ++i)
            printf("xi%i ", h_ColIndices[i]);
        printf("\n");

        double * d_my_solution;
        checkCudaErrors(cudaMalloc((void**)&d_my_solution, sizeof(double) * (horizon * StateShape)));

        std::cout<<"nnz "<<nnz<<std::endl;
////Test
        (cusolverSpDcsrlsvqr(solver_handle, N, 41, descrA, d_sparsematrix, d_RowIndices, d_ColIndices, d_dual, 0.000001, 0, d_my_solution, &singularity));

        double * my_solution = (double *)malloc(StateShape * horizon* sizeof(double));

        cudaMemcpy(my_solution, d_my_solution, sizeof(double) * (horizon * StateShape) ,cudaMemcpyDeviceToHost);

        std::cout<<"my_solution = "<<std::endl;
        for(int i = 0 ; i < StateShape*horizon ; i ++)
        {
            std::cout<<my_solution[i]<<" ";
        }
        std::cout<<std::endl;

////##########&&&&&&&&&&&&& EndTEST

        cusparseDnnz(handle, CUSPARSE_DIRECTION_ROW, Nrows, Ncols, descrA, dshared, lda, d_nnzPerVector, &nnz);

        (cudaMalloc(&d_A, nnz * sizeof(*d_A)));
        (cudaMalloc(&d_A_RowIndices, (Nrows + 1) * sizeof(*d_A_RowIndices) + 10));
        cudaMemset(d_A_RowIndices, 0, (Nrows + 1) * sizeof(*d_A_RowIndices) + 10);
        (cudaMalloc(&d_A_ColIndices, nnz * sizeof(*d_A_ColIndices)));

        (cusparseDdense2csr(handle, Nrows, Ncols, descrA, dshared, lda, d_nnzPerVector, d_A, d_A_RowIndices, d_A_ColIndices));

        std::cout<<"N = "<<N<<std::endl;
        std::cout<<"nnz = "<<nnz<<std::endl;
////Answer:
        (cusolverSpDcsrlsvqr(solver_handle, N, 41, descrA, d_sparsematrix, d_RowIndices, d_ColIndices, d_dual, 0.000001, 0, d_x, &singularity));

        //////////###########
       
        /////////##########@@@@@@@@@@@@ RealAnswer

        double *h_A = (double *)malloc(nnz * sizeof(*h_A));
        int *h_A_RowIndices = (int *)malloc((Nrows + 1) * sizeof(*h_A_RowIndices));
        int *h_A_ColIndices = (int *)malloc(nnz * sizeof(*h_A_ColIndices));
        (cudaMemcpy(h_A, d_A, nnz * sizeof(*h_A), cudaMemcpyDeviceToHost));
        (cudaMemcpy(h_A_RowIndices, d_A_RowIndices, (Nrows + 1) * sizeof(*h_A_RowIndices), cudaMemcpyDeviceToHost));
        (cudaMemcpy(h_A_ColIndices, d_A_ColIndices, nnz * sizeof(*h_A_ColIndices), cudaMemcpyDeviceToHost));

        double * h_x = (double * )malloc(sizeof(double) * StateShape * horizon);

        std::cout << "realAnswer : " << std::endl;
        (cudaMemcpy(h_x, d_x, sizeof(double) * StateShape * horizon, cudaMemcpyDeviceToHost));
        // for (int i = 0; i < nnz; ++i)
        //     printf("A[%i] = %f ", i, h_A[i]);
        // printf("\n");
        // printf("h_A_RowIndices] = \n ");
        for (int i = 0; i < (Nrows + 1); ++i)
            printf("%i ", h_A_RowIndices[i]);
        printf("\n");
     
          printf("h_A_ColIndices = \n ");
        for (int i = 0; i < 41; ++i)
            printf("la%i ", h_A_ColIndices[i]);
        printf("\n");

        for (int i = 0; i < StateShape * horizon; ++i)
            printf("%f ", h_x[i]);
        printf("\n");
        

        /////////##########@@@@@@@@@@@@

        Second_solve_kernel<<<1, horizon + 1>>>(solver_gpu, d_my_solution);
        checkCudaErrors(cudaDeviceSynchronize());
    }

    checkCudaErrors(cudaMemcpy(cache, solver_gpu, sizeof(TinyCache) * (horizon + 1), cudaMemcpyDeviceToHost));

    std::cout<<"my answer"<<std::endl;
    debug(cache);

    checkCudaErrors(cudaMemcpy(shared, dshared, sizeof(SharedMatrix), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(bigDual, d_dual, sizeof(TempBigDual), cudaMemcpyDeviceToHost));
}


