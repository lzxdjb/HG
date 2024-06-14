#pragma once

#include <iostream>
#include <vector>
#include <Eigen/Dense>

using namespace Eigen;

const int StateShape = 3;
const int ControlShape = 2;
const double T = 1.0;
const int horizon = 3;

class StateCostEdge;
class ControlCostEdge;
class EqualityEdge;

class StateVertex {
public:
    VectorXd state;
    std::vector<StateCostEdge> Cost;
    std::vector<EqualityEdge> Equality;
    std::vector<MatrixXd> Jacobian;
    std::vector<MatrixXd> Gradient;
    std::vector<MatrixXd> Hessian;
    std::vector<MatrixXd> EqualityTensorList;
    MatrixXd EqualityTensor;

    StateVertex(VectorXd state);
    void add_CostEdge(StateCostEdge edge);
    void add_EqualityEdge(EqualityEdge edge);
    void getEquality();
    void getJacobian();
    void getGradient();
    void getHessian();
    void update(VectorXd vector, double learning_rate);
};

class ControlVertex {
public:
    VectorXd control;
    std::vector<ControlCostEdge> Cost;
    std::vector<EqualityEdge> Equality;
    std::vector<MatrixXd> Jacobian;
    std::vector<MatrixXd> Gradient;
    std::vector<MatrixXd> Hessian;
    std::vector<MatrixXd> EqualityTensorList;

    ControlVertex(VectorXd control);
    void add_CostEdge(ControlCostEdge edge);
    void add_EqualityEdge(EqualityEdge edge);
    void getJacobian();
    void getGradient();
    void getHessian();
    void update(VectorXd vector, double learning_rate);
};

class StateCostEdge {
public:
    StateVertex *state0;
    MatrixXd Q;
    double cost;

    StateCostEdge(StateVertex *state0, MatrixXd Q);

    MatrixXd CostGradient();
    MatrixXd CostHessian();
};

class ControlCostEdge {
public:
    ControlVertex *control;
    MatrixXd R;
    double cost;

    ControlCostEdge(ControlVertex *control, MatrixXd R);
    MatrixXd CostGradient();
    MatrixXd CostHessian();
};

class EqualityEdge {
public:
    VectorXd state0;
    StateVertex * state1;
    ControlVertex  * control;
    std::vector<MatrixXd> equality;
    MatrixXd equalityTensor;

    EqualityEdge(VectorXd state0, StateVertex *state1, ControlVertex *control);
    MatrixXd getEquality();
    MatrixXd constraint_jacobian(VectorXd state, int i);
};

class NewtonMethod {
public:
    VectorXd init;
    MatrixXd Q;
    MatrixXd R;
    MatrixXd A;
    double T;
    int horizon;

    std::vector<StateVertex * > States;
    std::vector<ControlVertex * > Controls;
    std::vector<StateCostEdge> StateCost;
    std::vector<ControlCostEdge> ControlCost;
    std::vector<EqualityEdge> EqualityEdges;

    NewtonMethod(VectorXd x_initial, VectorXd x_final, MatrixXd Q, MatrixXd R, MatrixXd A, double T, int horizon);
    void debug_output();
    void setup_problem(VectorXd x_initial, VectorXd x_final);
    void ProblemGetEquality(int i);
    void ProblemGetJB(int i);
    void ProblemGetGradient(int i);
    void ProblemGetHessian(int i);
    MatrixXd GetFinalMatrixInverse(MatrixXd StateHessian, MatrixXd ControlHessian, MatrixXd StateJacobian, MatrixXd ControlJacobian);
    VectorXd GetFinalColumn(MatrixXd StateGradient, MatrixXd ControlGradient, MatrixXd StateConstraint);
    // void train();
};

