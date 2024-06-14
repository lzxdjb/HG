#include "newton_method.h"

int main() {
    VectorXd x_initial(StateShape);
    x_initial << 0, 0, 0;
    VectorXd x_final(StateShape);
    x_final << 1.5, 1.5, 0;
    MatrixXd Q = MatrixXd::Zero(StateShape, StateShape);
    Q(0, 0) = 1.0;
    Q(1, 1) = 5.0;
    Q(2, 2) = 0.1;

    MatrixXd R = MatrixXd::Zero(ControlShape, ControlShape);
    R(0, 0) = 0.5;
    R(1, 1) = 0.05;
    MatrixXd A = MatrixXd::Identity(StateShape, StateShape);

    NewtonMethod newton_method(x_initial, x_final, Q, R, A, T, horizon);
    newton_method.setup_problem(x_initial, x_final);
    newton_method.debug_output();

    // newton_method.train();

    return 0;
}
