#include "head.cuh"

void debug(TinyCache *solvergpu , SharedMatrix * dshared)
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

        // std::cout << "LowerLeftDown = \n" << solvergpu[i].LowerLeftDown1 << std::endl;

        // std::cout << "LowerLeftDown = \n" << solvergpu[i].LowerLeftDown2 << std::endl;

        // std::cout << "debug = \n" << solvergpu[i].debug << std::endl;

        // std::cout << "shared = \n" << &shared << std::endl;
       

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

__global__ void solve_kernel(TinyCache *solver_gpu , SharedMatrix * dshared)
{

    int idx = threadIdx.x;
    if (idx < horizon)
    {
       __shared__ SharedMatrix shared;


        for(int i = 0 ; i < horizon * StateShape ; i++)
        {
            for(int j = 0 ; j < horizon ; j ++)
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

        // while (1)

        for (int i = 0; i < 1; i++)
        {

            Hessian = GetHessian(Q, R);
            equality = GetEquality(solver_gpu[idx].state1, solver_gpu[idx].control, solver_gpu[idx].initial_state);

            if (idx != horizon - 1)
            {                             
                //  printf("asdfasd");

                if (idx == 0)
                {
                    JB1 = GetJB1(solver_gpu[idx].state1 , solver_gpu[idx].initial_state);
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

        /////////
            temp temp1 , temp2 , temp3 , temp4;


            if(idx != horizon - 1)
            {

            temp1 = - LowerLeftDown1.lazyProduct(JB1.transpose());
            temp2 = - LowerLeftDown1.lazyProduct(JB2.transpose());
             temp3 = - LowerLeftDown2.lazyProduct(JB1.transpose());
            temp4 = - LowerLeftDown2.lazyProduct(JB2.transpose());

            mycopy(&shared , temp1 , temp2 , temp3 , temp4, idx);

              
            }
            else{
                temp1 = - LowerLeftDown1.lazyProduct(JB1.transpose());
                mycopy2(&shared , temp1 , idx);
            }

            



            
         


            // finalMatrix = GetFinalMatrix(Hessian, JB);

            // finalColumn = GetFinalColumn(Allgradient, equality);

            // solver_gpu[idx].OriginalMatrix = finalMatrix;
            // solver_gpu[idx].OriginalColumn = finalColumn;

            // L = FinalMatrix::Identity(finalMatrix.rows(), finalMatrix.cols());

            // RowElimination(&finalMatrix, &L, 0, 0, 5, 0);
            // RowElimination(&finalMatrix, &L, 1, 1, 6, 1);
            // RowElimination(&finalMatrix, &L, 2, 2, 7, 2);
            // RowElimination(&finalMatrix, &L, 4, 4, 7, 4);
            // // FinalMatrix tem =  finalMatrix;

            // RowElimination(&finalMatrix, &L, 3, 3, 5, 3);
            // RowElimination(&finalMatrix, &L, 3, 3, 6, 3);
            // RowElimination(&finalMatrix, &L, 5, 5, 6, 5);

            // solve1(&L, &finalMatrix, &varible1, &varible2, &finalColumn);
            // solver_gpu[idx].state1.segment(0, StateShape) += learning_rate * varible2.segment(0, StateShape);

            // solver_gpu[idx].control.segment(0, ControlShape) += learning_rate * varible2.segment(StateShape, ControlShape);

            // // if (pivot == idx)
            // // {
            // //     norm = 10;
            // // }
            // // else

            // norm = varible2.segment(0, ControlShape + StateShape).norm();
            // // }

            // printf("norm %f \n", norm);
            // printf("pivot =  %d\n", pivot);
            // printf("idx %d \n", idx);

            // if (norm < 1e-8 && horizon - 1 == idx)
            // {

            //     printf("@@@@@@@@@@@@@@ idx = %d , pivot = %d \n", idx, pivot);
            //     for (int i = pivot + 1; i < horizon; i++)
            //     {

            //         solver_gpu[i].initial_state.segment(0 , StateShape) = solver_gpu[pivot].state1.segment(0 , StateShape);

            //     }

            //     pivot++;
            // }
            // __syncthreads();

            // // atomicAdd(&pivot, 1);  // Only one thread in the warp updates the pivot
            // if (pivot - 1 == idx)
            // {
            //     printf("jbjbjbjbjbjbj pivot = %d , idx = %d\n", pivot, idx);

            //     break;
            // }
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
        solver_gpu[idx].LowerLeftDown1 = LowerLeftDown1; 

        solver_gpu[idx].LowerLeftDown2 = LowerLeftDown2;

        for(int i = 0 ; i < horizon * StateShape ; i ++)
        {
            for (int j = 0 ; j < horizon * StateShape ; j ++)
            {
                dshared->row(i)[j] = shared.row(i)[j];
            }
        }
   
    }
}

void tiny_solve_cuda(TinyCache *cache , SharedMatrix * shared)
{
    TinyCache *solver_gpu;

    SharedMatrix * dshared ; 
    

    // debug(cache);
    checkCudaErrors(cudaMalloc((void **)&solver_gpu, sizeof(TinyCache) * horizon));
    checkCudaErrors(cudaMemcpy(solver_gpu, cache, sizeof(TinyCache) * horizon, cudaMemcpyHostToDevice));

    /////
    checkCudaErrors(cudaMalloc((void **)&dshared, sizeof(SharedMatrix)));
    checkCudaErrors(cudaMemcpy(dshared, shared, sizeof(SharedMatrix), cudaMemcpyHostToDevice));

/////

    solve_kernel<<<1, horizon>>>(solver_gpu , dshared);
    checkCudaErrors(cudaDeviceSynchronize());



    checkCudaErrors(cudaMemcpy(cache, solver_gpu, sizeof(TinyCache) * horizon, cudaMemcpyDeviceToHost));

/////
    checkCudaErrors(cudaMemcpy(shared, dshared, sizeof(SharedMatrix), cudaMemcpyDeviceToHost));

    debug(cache , dshared);
}
