#include "head.cuh"

void debug(TinyCache *solvergpu)
{
    for (int i = 0; i < horizon; i++)
    {
        std::cout << "state_vectors = " << solvergpu[i].state1 << std::endl;
        std::cout << "control_vectors = " << solvergpu[i].control << std::endl;

        // std::cout << "gradient = " << solvergpu[i].gradient << std::endl;
        // std::cout << "Hessian = " << solvergpu[i].Hessian << std::endl;

        // std::cout << "equality = " << solvergpu[i].equality << std::endl;
        // std::cout << "JB = " << solvergpu[i].JB << std::endl;

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
    }
}

__global__ void solve_kernel(TinyCache *solver_gpu)
{

    int idx = threadIdx.x;
    __shared__ SharedDirection SharedDirection;
    __shared__ SharedState SharedState;
    __shared__ int pivot;
    pivot = 0;
    if (idx < horizon)
    {
        QCost Q = solver_gpu[idx].Q;
        RCost R = solver_gpu[idx].R;

        gradient Allgradient;
        Hessian Hessian;
        equality equality;
        JB JB;

        FinalMatrix finalMatrix;
        FinalColumn finalColumn;

        varible varible1;
        varible varible2;

        FinalMatrix L;
        // solve2();

        double learning_rate = 1;

        // SharedState.row(idx).segment(0, StateShape) = solver_gpu[idx].state1.segment(0, StateShape).transpose();

        // SharedState.row(idx).segment(0, StateShape) = solver_gpu[idx].state1.segment(0, StateShape).transpose();
        // printf("temp %f" , SharedState.row(idx)[0]);
        // printf("temp %f" , SharedState.row(idx)[1]);
        // printf("temp %f" , SharedState.row(idx)[2]);
        tinytype norm;
        while (1)

        // for (int i = 0; i < 20; i++)
        {
            //     int local_pivot = 0;

            // local_pivot = __shfl_sync(0xFFFFFFFF, local_pivot, 0);

            //  printf("tempzzz %f" , varible2.row(0)[0]);
            //  printf("tempzzz %f" , varible2.row(1)[0]);
            //  printf("tempzzz %f" , varible2.row(2)[0]);
            // SharedState.row(idx).segment(0, StateShape) += varible2.segment(0, StateShape);

            Hessian = GetHessian(Q, R);
            equality = GetEquality(solver_gpu[idx].state1, solver_gpu[idx].control, solver_gpu[idx].initial_state);
            JB = GetJB(solver_gpu[idx].state1, solver_gpu[idx].control, solver_gpu[idx].initial_state);
            Allgradient = GetGradient(solver_gpu[idx].state1, solver_gpu[idx].control, solver_gpu[idx].final_state, Q, R);

            finalMatrix = GetFinalMatrix(Hessian, JB);

            finalColumn = GetFinalColumn(Allgradient, equality);

            solver_gpu[idx].OriginalMatrix = finalMatrix;
            solver_gpu[idx].OriginalColumn = finalColumn;

            L = FinalMatrix::Identity(finalMatrix.rows(), finalMatrix.cols());

            RowElimination(&finalMatrix, &L, 0, 0, 5, 0);
            RowElimination(&finalMatrix, &L, 1, 1, 6, 1);
            RowElimination(&finalMatrix, &L, 2, 2, 7, 2);
            RowElimination(&finalMatrix, &L, 4, 4, 7, 4);
            // FinalMatrix tem =  finalMatrix;

            RowElimination(&finalMatrix, &L, 3, 3, 5, 3);
            RowElimination(&finalMatrix, &L, 3, 3, 6, 3);
            RowElimination(&finalMatrix, &L, 5, 5, 6, 5);

            solve1(&L, &finalMatrix, &varible1, &varible2, &finalColumn);
            solver_gpu[idx].state1.segment(0, StateShape) += learning_rate * varible2.segment(0, StateShape);

            solver_gpu[idx].control.segment(0, ControlShape) += learning_rate * varible2.segment(StateShape, ControlShape);

            // if (pivot == idx)
            // {
            //     norm = 10;
            // }
            // else

            norm = varible2.segment(0, ControlShape + StateShape).norm();
            // }

            printf("norm %f \n", norm);
            printf("pivot =  %d\n", pivot);
            printf("idx %d \n", idx);

    
            if (norm < 1e-8 && horizon - 1 == idx)
            {

                printf("@@@@@@@@@@@@@@ idx = %d , pivot = %d \n", idx, pivot);
                for (int i = pivot + 1; i < horizon; i++)
                {
                  

                    solver_gpu[i].initial_state.segment(0 , StateShape) = solver_gpu[pivot].state1.segment(0 , StateShape);

               
                }
            
                pivot++;
            }
            __syncthreads();

            // atomicAdd(&pivot, 1);  // Only one thread in the warp updates the pivot
            if (pivot - 1 == idx)
            {
                printf("jbjbjbjbjbjbj pivot = %d , idx = %d\n", pivot, idx);

                break;
            }

        }

        solver_gpu->equality = equality;
        solver_gpu->JB = JB;
        solver_gpu->Hessian = Hessian;
        solver_gpu->gradient = Allgradient;
        solver_gpu->FinalMatrix = finalMatrix;
        solver_gpu->FinalColumn = finalColumn;
        solver_gpu->L = L;

        solver_gpu->varible1 = varible1;
        solver_gpu->varible2 = varible2;
    }
}

void tiny_solve_cuda(TinyCache *cache)
{
    TinyCache *solver_gpu;

    // debug(cache);
    checkCudaErrors(cudaMalloc((void **)&solver_gpu, sizeof(TinyCache) * horizon));
    checkCudaErrors(cudaMemcpy(solver_gpu, cache, sizeof(TinyCache) * horizon, cudaMemcpyHostToDevice));

    solve_kernel<<<1, horizon>>>(solver_gpu);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(cache, solver_gpu, sizeof(TinyCache) * horizon, cudaMemcpyDeviceToHost));
    debug(cache);
}
