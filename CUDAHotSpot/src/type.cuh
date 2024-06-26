#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>

typedef double tinytype;  // should be double if you want to generate code

// #define NSTATES 12
// #define NINPUTS 4

const int StateShape = 3;
const int ControlShape = 2;
const double T = 1;
const int horizon = 2;

using Eigen::Matrix;



    typedef Matrix<tinytype, StateShape , 1> state;
    typedef Matrix<tinytype, ControlShape , 1> control;
    // typedef Matrix<tinytype, StateShape, 1> initial;
    typedef Matrix<tinytype, StateShape, 1> final_state;

    typedef Matrix<tinytype, (StateShape * 2 + ControlShape), 1> varible;
    typedef Matrix<tinytype, StateShape + ControlShape, StateShape + ControlShape> Hessian;
    typedef Matrix<tinytype, StateShape + ControlShape, 1> gradient;
    typedef Matrix<tinytype, StateShape, 1> equality;

    typedef Matrix<tinytype,  StateShape, StateShape + ControlShape> JB; 
    typedef Matrix<tinytype,  StateShape, StateShape > StateJB;   
    typedef Matrix<tinytype,  StateShape, ControlShape > ControlJB;   


    typedef Matrix<tinytype,  StateShape, StateShape > QCost;  

    typedef Matrix<tinytype,  ControlShape, ControlShape> RCost;  

    typedef Matrix<tinytype,  StateShape * 2 + ControlShape,   StateShape * 2 + ControlShape > FinalMatrix;   
    typedef Eigen::SparseMatrix<tinytype> SparseFinalMatrix;

    typedef Matrix<tinytype,   StateShape * 2 + ControlShape , 1 > FinalColumn;   
    

    typedef Matrix<tinytype,  horizon, StateShape * 2 + ControlShape > SharedDirection;   
    typedef Matrix<tinytype,  horizon, StateShape > SharedState; 

    typedef struct
    {
        state state1;
        control control;
        state initial_state;
        state final_state;
        QCost Q;
        RCost R;
        // varible varible;
        gradient gradient;
        equality equality;
        Hessian Hessian ;
        JB JB;

        FinalMatrix OriginalMatrix;
        FinalColumn OriginalColumn;

        FinalColumn varible1;
        FinalColumn varible2;


        FinalMatrix L;
        FinalMatrix FinalMatrix;
        FinalColumn FinalColumn;


       

    } TinyCache;
