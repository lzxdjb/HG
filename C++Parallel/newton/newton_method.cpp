#include "newton_method.h"
#include <iostream>
using namespace std;
// StateVertex methods
StateVertex::StateVertex(VectorXd state) : state(state) {}

void StateVertex::add_CostEdge(StateCostEdge edge) {
    // cout<<"asdfasdf"<<endl;
    Cost.push_back(edge);
    cout<<"Cost.size = "<<Cost.size();
}

void StateVertex::add_EqualityEdge(EqualityEdge edge) {
    Equality.push_back(edge);
}

StateCostEdge::StateCostEdge(StateVertex * state0, MatrixXd Q)
    : state0(state0), Q(Q)
{
    cost = (state0->state.transpose()) * Q * (state0->state);
    state0->add_CostEdge(*this);
}


ControlCostEdge::ControlCostEdge(ControlVertex *control,  MatrixXd R)
    :control(control), R(R)
{
    cost = (control->control.transpose()) * R * (control->control);;
    control->add_CostEdge(*this);
}

// void StateVertex::getEquality() {
//     EqualityTensorList.clear();

//     cout<<"edge = "  <<endl;
//     for (auto& edge : Equality) {

//         cout<<"edge = "<< edge <<endl;
//         EqualityTensorList.push_back(edge);
//     }
//     exit(0);

//     EqualityTensor = MatrixXd(EqualityTensorList.size(), 1);
//     for (size_t i = 0; i < EqualityTensorList.size(); ++i) {
//         EqualityTensor(i, 0) = EqualityTensorList[i](0, 0);
//     }
// }

// void StateVertex::getJacobian() {
//     Jacobian.clear();
//     for (size_t i = 0; i < Equality.size(); ++i) {
//         Jacobian.push_back(Equality[i]);
//     }
// }

// void StateVertex::getGradient() {
//     Gradient.clear();
//     for (auto& edge : Cost) {
//         Gradient.push_back(edge);
//     }
// }

// void StateVertex::getHessian() {
//     Hessian.clear();
//     for (auto& edge : Cost) {
//         Hessian.push_back(edge);
//     }
// }

// void StateVertex::update(VectorXd vector, double learning_rate) {
//     state += vector * learning_rate;
// }

// ControlVertex methods
ControlVertex::ControlVertex(VectorXd control) : control(control) {};

void ControlVertex::add_CostEdge(ControlCostEdge edge) {
    Cost.push_back(edge);
}

void ControlVertex::add_EqualityEdge(EqualityEdge edge) {
    Equality.push_back(edge);
}

// void ControlVertex::getJacobian() {
//     Jacobian.clear();
//     for (size_t i = 0; i < Equality.size(); ++i) {
//         Jacobian.push_back(Equality[i]);
//     }
// }

// void ControlVertex::getGradient() {
//     Gradient.clear();
//     for (auto& edge : Cost) {
//         Gradient.push_back(edge);
//     }
// }

// void ControlVertex::getHessian() {
//     Hessian.clear();
//     for (auto& edge : Cost) {
//         Hessian.push_back(edge);
//     }
// }

// void ControlVertex::update(VectorXd vector, double learning_rate) {
//     control += vector * learning_rate;
// }

// StateCostEdge methods


// MatrixXd StateCostEdge::CostGradient() {
//     return 2 * Q * (state0->state);
// }

// MatrixXd StateCostEdge::CostHessian() {
//     return 2 * Q;
// }

// // ControlCostEdge methods
// ControlCostEdge::ControlCostEdge(ControlVertex* control, MatrixXd R) : control(control), R(R) {
//     cost = control->control.transpose() * R * control->control;
// }

// MatrixXd ControlCostEdge::CostGradient() {
//     return 2 * R * control->control;
// }

// MatrixXd ControlCostEdge::CostHessian() {
//     return 2 * R;
// }

// EqualityEdge methods
EqualityEdge::EqualityEdge(VectorXd state0, StateVertex *state1, ControlVertex * control) 
    : state0(state0), state1(state1), control(control) {
    state1->add_EqualityEdge(*this);
    control->add_EqualityEdge(*this);
}

// MatrixXd EqualityEdge::getEquality() {
//     equality.clear();
//     equality.push_back(MatrixXd::Constant(1, 1, state1->state[0] - state0[0] - T * cos(state0[2]) * control->control[0]));
//     equality.push_back(MatrixXd::Constant(1, 1, state1->state[1] - state0[1] - T * sin(state0[2]) * control->control[0]));
//     equality.push_back(MatrixXd::Constant(1, 1, state1->state[2] - state0[2] - T * control->control[1]));
    
//     equalityTensor = MatrixXd(equality.size(), 1);
//     for (size_t i = 0; i < equality.size(); ++i) {
//         equalityTensor(i, 0) = equality[i](0, 0);
//     }
//     return equalityTensor;
// }

// MatrixXd EqualityEdge::constraint_jacobian(VectorXd state, int i) {
//     if (state == control->control) {
//         MatrixXd jacobian(3, 2);
//         jacobian << -cos(state0[2]), 0,
//                     -sin(state0[2]), 0,
//                     0, -1;
//         return jacobian;
//     } else {
//         return MatrixXd::Identity(3, 3);
//     }
// }

// NewtonMethod methods
NewtonMethod::NewtonMethod(VectorXd x_initial, VectorXd x_final, MatrixXd Q, MatrixXd R, MatrixXd A, double T, int horizon)
    : init(x_initial), Q(Q), R(R), A(A), T(T), horizon(horizon) {}

void NewtonMethod::debug_output() {
    std::cout << "Vertices and their associated edges:\n";
    std::cout << "for states\n";
    for (size_t i = 0; i < States.size(); ++i) {
        std::cout << "#######cost :\n";
        std::cout << "States " << i << ":\n";
        // cout << "size = " << States[i]->Cost.size() << ":\n";
        // exit(0);
    // }
    
        for (auto& edge : States[i]->Cost) {
            // StateCostEdge edge = States[i]->Cost[0];
            std::cout << "  Edge connecting to state " << edge.state0.state << "\n";
            cout<<"asdfasdf"<<endl;
        // }
        std::cout << "#######contrain :\n";
        std::cout << "States " << i << ":\n";
        }
    //     // for (auto& edge : States[i].Equality) {
    //         auto& edge1 = States[i].Equality[0];
    //         std::cout << "  Edge connecting to state " << edge1.state1.state << "and control = " << edge1.control.control;
    //     // }
    // }

    // std::cout << "for control\n";
    // for (size_t i = 0; i < Controls.size(); ++i) {
    //     std::cout << "#######cost :\n";
    //     std::cout << "Controls " << i << ":\n";
    //     for (auto& edge : Controls[i].Cost) {
    //         std::cout << "  Edge connecting to control " << edge.control.control << "\n";
    //     }
    //     std::cout << "#######contrain :\n";
    //     std::cout << "Controls " << i << ":\n";
    //     for (auto& edge : Controls[i].Equality) {
    //         std::cout << "  Edge connecting to state " << edge.state1.state << "and control = " << edge.control.control;
    //     }
    // }
    }
}

void NewtonMethod::setup_problem(VectorXd x_initial, VectorXd x_final) {
  

    for (int i = 0; i < horizon; ++i)
    {
        VectorXd control = VectorXd::Zero(ControlShape);
        ControlVertex * control_vertex = new ControlVertex(control);

        Controls.push_back(control_vertex);

        VectorXd state = VectorXd::Zero(StateShape);
        StateVertex * state_vertex = new StateVertex(state);

        States.push_back(state_vertex);

    }

    for (int i = 0; i < horizon; ++i) 
    {
        EqualityEdge equality_edge(x_initial, States[i], Controls[i]);

        EqualityEdges.push_back(equality_edge);

    }
    for(int i = 0 ; i < horizon ; i ++)
    {
      
        StateCostEdge state_cost(States[i], Q);
        StateCost.push_back(state_cost);

        ControlCostEdge control_cost(Controls[i], R);
        ControlCost.push_back(control_cost);
     
    }
}

// void NewtonMethod::ProblemGetEquality(int i) {
//     States[i].getEquality();
//     Controls[i].getJacobian();
// }

// void NewtonMethod::ProblemGetJB(int i) {
//     States[i].getJacobian();
//     Controls[i].getJacobian();
// }

// void NewtonMethod::ProblemGetGradient(int i) {
//     States[i].getGradient();
//     Controls[i].getGradient();
// }

// void NewtonMethod::ProblemGetHessian(int i) {
//     States[i].getHessian();
//     Controls[i].getHessian();
// }

// MatrixXd NewtonMethod::GetFinalMatrixInverse(MatrixXd StateHessian, MatrixXd ControlHessian, MatrixXd StateJacobian, MatrixXd ControlJacobian) {
//     int shape = StateHessian.rows() + ControlHessian.rows();
//     MatrixXd MatrixInverse = MatrixXd::Zero(shape, shape);
//     MatrixXd StateMatrix = StateHessian + StateJacobian.transpose() * StateJacobian;
//     MatrixXd ControlMatrix = ControlHessian + ControlJacobian.transpose() * ControlJacobian;
//     MatrixInverse.block(0, 0, StateMatrix.rows(), StateMatrix.cols()) = StateMatrix.inverse();
//     MatrixInverse.block(StateMatrix.rows(), StateMatrix.cols(), ControlMatrix.rows(), ControlMatrix.cols()) = ControlMatrix.inverse();
//     return MatrixInverse;
// }

// VectorXd NewtonMethod::GetFinalColumn(MatrixXd StateGradient, MatrixXd ControlGradient, MatrixXd StateConstraint) {
//     int shape = StateGradient.rows() + ControlGradient.rows();
//     VectorXd finalColumn(shape);
//     finalColumn << StateGradient, ControlGradient;
//     return finalColumn;
// }

// void NewtonMethod::train() {
//     double learning_rate = 0.1;
//     for (int iter = 0; iter < 10; ++iter) {
//         for (int i = 0; i < horizon; ++i) {
//             ProblemGetEquality(i);
//             ProblemGetJB(i);
//             ProblemGetGradient(i);
//             ProblemGetHessian(i);
            
//             std::cout<<States[i].EqualityTensorList[0]<<std::endl;

//             // std::cout<<States[i].
//             // exit(0);

//             MatrixXd StateHessian = States[i].Hessian[0];
//             MatrixXd ControlHessian = Controls[i].Hessian[0];
//             MatrixXd StateJacobian = States[i].Jacobian[0];
//             MatrixXd ControlJacobian = Controls[i].Jacobian[0];
//             MatrixXd StateGradient = States[i].Gradient[0];
//             MatrixXd ControlGradient = Controls[i].Gradient[0];
            
//             // std::cout<<StateHessian<<std::endl;
//             // std::cout<<ControlHessian<<std::endl;
//             // std::cout<<StateJacobian<<std::endl;
//             // std::cout<<ControlJacobian<<std::endl;
//             // std::cout<<StateGradient<<std::endl;
//             // std::cout<<ControlGradient<<std::endl;
//             exit(1);



//             MatrixXd MatrixInverse = GetFinalMatrixInverse(StateHessian, ControlHessian, StateJacobian, ControlJacobian);
//             VectorXd finalColumn = GetFinalColumn(StateGradient, ControlGradient, States[i].EqualityTensor);

//             VectorXd update_vector = MatrixInverse * finalColumn;
//             States[i].update(update_vector.head(StateShape), learning_rate);
//             Controls[i].update(update_vector.tail(ControlShape), learning_rate);
//         }
//     }
// }
