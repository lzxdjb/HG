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

       
        // while (1)

        // for (int i = 0; i < 20; i++)
        
            

            Hessian = GetHessian(Q, R);
            equality = GetEquality(solver_gpu[idx].state1, solver_gpu[idx].control, solver_gpu[idx].initial_state);
            JB = GetJB(solver_gpu[idx].state1, solver_gpu[idx].control, solver_gpu[idx].initial_state);
            Allgradient = GetGradient(solver_gpu[idx].state1, solver_gpu[idx].control, solver_gpu[idx].final_state, Q, R);

            finalMatrix = GetFinalMatrix(Hessian, JB);

            finalColumn = GetFinalColumn(Allgradient, equality);

           

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
