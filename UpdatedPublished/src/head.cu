#include "head.cuh"
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
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

        // std::cout << "FirstVarible = \n"<< solvergpu[i].FirstVarible << std::endl;

        // std::cout << "FirstDual = \n" << solvergpu[i].FirstDual << std::endl;

        //  std::cout << "soltuion temp = \n"<< solvergpu[i].solutionTemp << std::endl;

        // std::cout << "convergence = \n"<< solvergpu[i].convergence << std::endl;
        std::cout<<"cache1 \n= " <<solvergpu[i].cache1<<std::endl;
        std::cout<<"cache2 \n= " <<solvergpu[i].cache2<<std::endl;
        std::cout<<"cache3 \n= " <<solvergpu[i].cache3<<std::endl;
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


            for(int i = 1 ; i < StateShape + 1 ; i ++)
            {
                solver_gpu[idx].h_A_RowIndices.row(i)[0] += index[idx - 1];
            }
             RowIndicescopyToPointer(solver_gpu[idx].h_A_RowIndices , RowIndices , 1 , StateShape , (idx- 1) * StateShape + 1);
        }

        if(idx < 3)
        {
            ColIndicescopyToPointer(solver_gpu[idx].h_A_ColIndices , ColIndices , 0 , index[idx] - index[idx - 1] , index[idx - 1]);
        }
        else{

            for(int i = 0 ; i < index[idx] - index[idx - 1] ; i ++)
            {
                solver_gpu[idx].h_A_ColIndices.row(i)[0] += StateShape * (idx - 2);
            }

            ColIndicescopyToPointer(solver_gpu[idx].h_A_ColIndices , ColIndices , 0 , index[idx] - index[idx - 1] , index[idx - 1]);
        }

    }
}

__global__ void solve_kernel(TinyCache *solver_gpu, double *bigDual, QCost Q, RCost R, state init_state, state final_state , int* d_index)
{

    int idx = threadIdx.x;
    // if (idx < horizon + 1 && idx > 0)
    {

        gradient Allgradient;
        Hessian Hessian;
        equality equality;

        JB JB1;
        JB JB2;


        JB LowerLeftDown1;
        JB LowerLeftDown2;

        LowerLeftDown1.setZero();
        LowerLeftDown2.setZero();

        Hessian = GetHessian(Q, R);

    
        equality = GetEquality(solver_gpu[idx + 1].state1, solver_gpu[idx + 1].control, solver_gpu[idx].state1);

        solver_gpu[idx + 1].equality = equality;
        

        // if (idx + 1 != horizon)
        {

            JB1 = GetJB1(solver_gpu[idx + 1].state1, solver_gpu[idx].state1);
            JB2 = GetJB2(solver_gpu[idx + 1].state1, solver_gpu[idx + 1].control);
            
        }

        // else
        // {
        // //     // printf("asdfasdf");
        //     JB1 = GetJB1(solver_gpu[idx + 1].state1, solver_gpu[idx].state1);
        //     JB2.setZero();
        // }

        Allgradient = GetGradient(solver_gpu[idx + 1].state1, solver_gpu[idx + 1].control, final_state, Q, R);

        solver_gpu[idx + 1].JB1 = JB1;
        solver_gpu[idx + 1].JB2 = JB2;
        solver_gpu[horizon].JB2.setZero();


        LowerLeftDown1 = solver_gpu[idx + 1].JB1.lazyProduct(PsedoInverse(Hessian));
        LowerLeftDown2 = solver_gpu[idx + 1].JB2.lazyProduct(PsedoInverse(Hessian));
        
        solver_gpu[idx + 1].LowerLeftDown2 = LowerLeftDown2;

        //// for debug
              solver_gpu[idx + 1].LowerLeftDown1 = LowerLeftDown1;
        ////


        if (idx == 0)
        {

 
            solver_gpu[idx + 1].cache1.topLeftCorner(StateShape , StateShape) += -solver_gpu[idx + 1].LowerLeftDown1.lazyProduct(solver_gpu[idx + 1].JB1.transpose());

            solver_gpu[idx + 1].cache2.topLeftCorner(StateShape , StateShape) += -solver_gpu[idx + 1].LowerLeftDown1.lazyProduct(solver_gpu[idx + 1].JB2.transpose());

            solver_gpu[idx + 2].cache1.topLeftCorner(StateShape , StateShape) += -solver_gpu[idx + 1].LowerLeftDown2.lazyProduct(solver_gpu[idx + 1].JB1.transpose());

            solver_gpu[idx + 2].cache2.topLeftCorner(StateShape , StateShape) += -solver_gpu[idx + 1].LowerLeftDown2.lazyProduct(solver_gpu[idx + 1].JB2.transpose());
        }
        
/// my new idea
    // A.block<3, 3>(0, 0) = A1;
    // A.block<3, 3>(0, 3) = A2;
    // A.block<3, 3>(0, 6) = A3;
    //     solver_gpu[idx + 1].bigcache.block<3, 3> = 

//////
        else if (idx < horizon - 1)
        {

            solver_gpu[idx + 1].cache2.topLeftCorner(StateShape , StateShape) += -solver_gpu[idx + 1].LowerLeftDown1.lazyProduct(solver_gpu[idx + 1].JB1.transpose());

            // printf("idx = %d " , idx);

           solver_gpu[idx + 1].cache3.topRightCorner(StateShape , StateShape) += -solver_gpu[idx + 1].LowerLeftDown1.lazyProduct(solver_gpu[idx + 1].JB2.transpose());

            solver_gpu[idx + 2].cache1.topLeftCorner(StateShape , StateShape) += -solver_gpu[idx + 1].LowerLeftDown2.lazyProduct(solver_gpu[idx + 1].JB1.transpose());

            solver_gpu[idx + 2].cache2.topRightCorner(StateShape , StateShape) += -solver_gpu[idx + 1].LowerLeftDown2.lazyProduct(solver_gpu[idx + 1].JB2.transpose());

        }
        else
        {
            solver_gpu[idx + 1].cache2.topRightCorner(StateShape , StateShape) += -solver_gpu[idx + 1].LowerLeftDown1.lazyProduct(solver_gpu[idx + 1].JB1.transpose());
          
    
        }
        // solver_gpu[idx].FirstVarible = -Allgradient;

        // __syncthreads();

        // state SolutionTemp;
        // if (idx == 1)
        // {
        //     SolutionTemp = LowerLeftDown1.lazyProduct(solver_gpu[idx].FirstVarible);
        // }
        // else
        // {
        //     // printf("asdfasdfas");
        //     SolutionTemp = LowerLeftDown1.lazyProduct(solver_gpu[idx].FirstVarible);

        //     SolutionTemp += solver_gpu[idx - 1].LowerLeftDown2.lazyProduct(solver_gpu[idx - 1].FirstVarible);
        // }

        // solver_gpu[idx].FirstDual = -equality - SolutionTemp;
       
        solver_gpu[idx + 1].gradient = Allgradient;

        solver_gpu[idx + 1].Hessian = Hessian;

        // solver_gpu[idx].nnz = dense_to_csr(solver_gpu[idx].cache1 , solver_gpu[idx].cache2 , solver_gpu[idx].cache3 ,&solver_gpu[idx].SparseCache , StateShape , StateShape * 3 , &solver_gpu[idx].h_A_RowIndices , &solver_gpu[idx].h_A_ColIndices);

        // d_index[idx] = solver_gpu[idx].nnz;

        // for (int i = 0; i < StateShape; i++)
        // {
        //     bigDual[(idx - 1) * StateShape + i] = solver_gpu[idx].FirstDual[i];
        // }
    }
}

__global__ void Second_solve_kernel(TinyCache *solver_gpu, double *d_x )
{
    int idx = threadIdx.x;
    double learning_rate = 1;
    // if (idx < horizon + 1 && idx > 0)
    {
        SecondPhaseCopy(&solver_gpu[idx].FirstDual, d_x, idx);

        convergence SolutionTemp;

        if (idx == horizon)
        {
            SolutionTemp = solver_gpu[idx].JB1.transpose().lazyProduct(solver_gpu[idx].FirstDual);
        }
        else if(idx > 0)
        {
            // printf("asdfasdfas");
            SolutionTemp = solver_gpu[idx].JB1.transpose().lazyProduct(solver_gpu[idx].FirstDual);

            SolutionTemp += solver_gpu[idx].JB2.transpose().lazyProduct(solver_gpu[idx + 1].FirstDual);
        }
        solver_gpu[idx].FirstVarible -= SolutionTemp;
        solver_gpu[idx].FirstVarible = PsedoInverse(solver_gpu[idx].Hessian).lazyProduct(solver_gpu[idx].FirstVarible);

        // solver_gpu[idx].convergence = SolutionTemp;

        solver_gpu[idx].state1 +=  solver_gpu[idx].FirstVarible.topLeftCorner(StateShape , 1) * learning_rate;

        solver_gpu[idx].control +=  solver_gpu[idx].FirstVarible.bottomRightCorner(ControlShape , 1) * learning_rate;


    }
}

void tiny_solve_cuda(TinyCache *cache,  tinytype *bigDual, QCost Q, RCost R, state init_state, state final_state){
    TinyCache *solver_gpu;
    double *d_dual; // FirstStageDual
    cudaMalloc(&d_dual, StateShape * horizon * sizeof(double));

    ////

    checkCudaErrors(cudaMalloc((void **)&solver_gpu, sizeof(TinyCache) * (horizon + 1) ));
    // printf("addr %d\n", solver_gpu);
    checkCudaErrors(cudaMemcpy(solver_gpu, cache, sizeof(TinyCache) * (horizon + 1), cudaMemcpyHostToDevice));

    cusparseHandle_t handle;
    (cusparseCreate(&handle));
    cusparseMatDescr_t descrA;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
 
    cusolverSpHandle_t solver_handle;
    cusolverSpCreate(&solver_handle);
    int singularity;

////////##########
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
    
    double * d_my_solution;
    checkCudaErrors(cudaMalloc((void**)&d_my_solution, sizeof(double) * (horizon * StateShape)));

//@@@@@@@@@@@
    int nnz ;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    bool flag = false;
    for (int i = 0; i < 1; i++)
    {

        solve_kernel<<<1, horizon>>>(solver_gpu, d_dual , Q, R, init_state, final_state , d_index);
        checkCudaErrors(cudaDeviceSynchronize());

        thrust::inclusive_scan(thrust::device_pointer_cast(d_index), thrust::device_pointer_cast(d_index) + horizon + 1, thrust::device_pointer_cast(d_index)); // in-place scan

        // checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaMemcpy(&nnz, d_index + horizon, 4, cudaMemcpyDeviceToHost));

////###########$$$$$$$$$$$$ TEST

        sparse_represent<<<1 , horizon + 1>>>(solver_gpu , d_index , d_sparsematrix , d_RowIndices , d_ColIndices);
     
////Test
        (cusolverSpDcsrlsvqr(solver_handle, StateShape * horizon, nnz, descrA, d_sparsematrix, d_RowIndices, d_ColIndices, d_dual, 0.000001, 0, d_my_solution, &singularity));

        // double * my_solution = (double *)malloc(StateShape * horizon* sizeof(double));


////##########&&&&&&&&&&&&& EndTEST
        Second_solve_kernel<<<1, horizon + 1>>>(solver_gpu, d_my_solution );
        checkCudaErrors(cudaDeviceSynchronize());
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time for sparse_represent: " << milliseconds * 1e-3 << " s" << std::endl;

    checkCudaErrors(cudaMemcpy(cache, solver_gpu, sizeof(TinyCache) * (horizon + 1), cudaMemcpyDeviceToHost));

    // std::cout<<"my answer"<<std::endl;
    debug(cache);

}


