#include "src/head.cuh"
using namespace std;

TinyCache cache[horizon];

int main()
{
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

        cache[i].L.Zero();
    }
    tiny_solve_cuda(cache);


    // std::cout<<"mytest =\n "<<cache[0].L.lazyProduct(cache[0].FinalMatrix) << "\n";


    Eigen::PartialPivLU<FinalMatrix> lu_decomp(cache[0].OriginalMatrix);

    FinalColumn x = lu_decomp.solve(cache[0].OriginalColumn);


    // std::cout << "Solution x: " << std::endl << x << std::endl;

    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> P = lu_decomp.permutationP();

    FinalMatrix Pd = P.toDenseMatrix().cast<tinytype>();


    FinalMatrix L = FinalMatrix::Identity(cache[0].FinalMatrix.rows(), cache[0].FinalMatrix.cols());
    L.triangularView<Eigen::StrictlyLower>() = lu_decomp.matrixLU();


    FinalMatrix U = lu_decomp.matrixLU().triangularView<Eigen::Upper>();

    // std::cout<<"test =\n "<<L.lazyProduct(U) << "\n";


    // Eigen::PartialPivLU<FinalMatrix> lu_d(L);
    // FinalColumn x1 = lu_d.solve(cache[0].OriginalColumn);

    // std::cout << "Solution x1: " << std::endl
    //           << x1 << std::endl;
   

    // Output L and U matrices
    // std::cout << "Matrix L: " << std::endl
    //           << L << std::endl;
    // std::cout << "Matrix U: " << std::endl
    //           << U << std::endl;
    // std::cout << "Permutation Matrix P: " << std::endl
    //           << Pd << std::endl;

  
}