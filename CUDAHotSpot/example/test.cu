
#include "src/head.cuh"

using namespace std;
int main()
{
    QCost Qtemp;
    RCost Rtemp;
    Qtemp << 1 , 0 , 0 ,
            0 , 1 , 0 , 
            0 , 0 , 1;
    Rtemp << 1 , 0,
         0 , 1;

    state state2;
    state2<<1,0,0;
    state initial_state;
    initial_state<<1,0,0;

    std::cout<<initial_state.block(0 , 0 , 3 , 1)<<endl;

    control control;
    control<<0 , 9;

Eigen::VectorXd Q_result = 2 * Qtemp * (state2 - initial_state);

Eigen::VectorXd R_result = 2 * Rtemp * control;

gradient temp ;
temp << Q_result , R_result;



}