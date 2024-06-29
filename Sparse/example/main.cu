#include "src/head.cuh"
using namespace std;

TinyCache cache[horizon];

// SharedMatrix shared;

// TempBigDual tempGibDual;
void debug1(double * shared)
{
    std::cout<<"shared\n";
    for(int i = 0 ; i < horizon * StateShape ; i ++)
    {
        for(int j = 0 ; j < horizon * StateShape ; j ++)
        {
            std::cout<<shared[i * horizon * StateShape + j]<<" ";
        }
        std::cout<<std::endl;
    }
}

void debug2(double * tempGibDual)
{
    std::cout<<"tempGibDual\n";
    for (int i = 0 ; i <  horizon * StateShape ; i ++)
    {
        std::cout<<tempGibDual[i]<<" ";
    }
}

int main()
{
    // shared.setZero();
    QCost Qtemp;
    RCost Rtemp;
    Qtemp << 1, 0, 0,
        0, 5, 0,
        0, 0, 0.1;
    Rtemp << 0.5, 0,
        0, 0.05;
    state final_state;
    final_state << 1.5, 1.5, 0;
    state init_state;
    init_state << 0 , 0 , 0;

    state temp ;
    temp<<1 , 1 , 1;
    control init_control;
    init_control<<1 , 1;

    for (int i = 0; i < horizon; i++)
    {
        cache[i].state1<<temp;
        cache[i].control<<init_control;
        cache[i].initial_state<< init_state;
        cache[i].final_state << final_state;
        cache[i].Q = Qtemp;
        cache[i].R = Rtemp;
        cache[i].debug.setZero();
    }

    tinytype *shared = (tinytype*)malloc(StateShape * horizon * StateShape * horizon * sizeof(tinytype));
    tinytype *tempGibDual = (tinytype *)malloc(StateShape * horizon * sizeof(tinytype));
    
    tiny_solve_cuda(cache , shared , tempGibDual);

    // debug1(shared);
    // debug2(tempGibDual);

  
}