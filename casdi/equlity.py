#!/usr/bin/env python
# -*- coding: utf-8 -*-

import casadi as ca
import casadi.tools as ca_tools

import numpy as np
import time
from draw import Draw_MPC_point_stabilization_v1


def shift_movement(T, t0, x0, u, x_f, f):
    f_value = f(x0, u[:, 0])
    control = u[:, 0]
    st = x0 + T * f_value.full()
    t = t0 + T
    u_end = np.concatenate((u[:, 1:], u[:, -1:]), axis=1)
    x_f = np.concatenate((x_f[:, 1:], x_f[:, -1:]), axis=1)

    return t, st, control , u_end, x_f


if __name__ == "__main__":
    T = 0.2  # sampling time [s]
    N = 3  # prediction horizon
    rob_diam = 0.3  # [m]
    v_max = 1e100
    omega_max = 1e100
    vertical_max = 1e100
    horizon_max = 1e100

    x = ca.MX.sym("x")
    y = ca.MX.sym("y")
    theta = ca.MX.sym("theta")
    states = ca.vertcat(x, y)
    states = ca.vertcat(states, theta)
    n_states = states.size()[0]

    v = ca.MX.sym("v")
    omega = ca.MX.sym("omega")
    controls = ca.vertcat(v, omega)
    n_controls = controls.size()[0]

    # rhs
    rhs = ca.vertcat(v * ca.cos(theta), v * ca.sin(theta))
    rhs = ca.vertcat(rhs, omega)

    # function
    f = ca.Function(
        "f", [states, controls], [rhs], ["input_state", "control_input"], ["rhs"]
    )

    # for MPC
    U = ca.MX.sym("U", n_controls, N)

    X = ca.MX.sym("X", n_states, N + 1)

    P = ca.MX.sym("P", n_states + n_states)

    # define
    Q = np.array([[1.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.1]])
    R = np.array([[0.5, 0.0], [0.0, 0.05]])
    # cost function
    obj = 0  #### cost
    g = []  # equal constrains
    g.append(X[:, 0] - P[:3])
    for i in range(N):
        obj = (
            obj
            + ca.mtimes([(X[:, i] - P[3:]).T, Q, X[:, i] - P[3:]])
            + ca.mtimes([U[:, i].T, R, U[:, i]])
        )
        x_next_ = f(X[:, i], U[:, i]) * T + X[:, i]
        g.append(X[:, i + 1] - x_next_)

    opt_variables = ca.vertcat(ca.reshape(U, -1, 1), ca.reshape(X, -1, 1))

    nlp_prob = {"f": obj, "x": opt_variables, "p": P, "g": ca.vertcat(*g)}

    print("Objective function:")
    print(obj)

    print("\nConstraints:")
    for constraint in g:
        print(constraint)

    print("#############")
    opts_setting = {
        "ipopt.max_iter": 100,
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.acceptable_tol": 1e-8,
        "ipopt.acceptable_obj_change_tol": 1e-6,
    }

    solver = ca.nlpsol("solver", "ipopt", nlp_prob, opts_setting)

    lbg = 0
    ubg = 0
    lbx = []
    ubx = []
    for _ in range(N):
        lbx.append(-v_max)
        lbx.append(-omega_max)
        ubx.append(v_max)
        ubx.append(omega_max)
    for _ in range(
        N + 1
    ):  # note that this is different with the method using structure
        lbx.append(-vertical_max)
        lbx.append(-horizon_max)
        lbx.append(-np.inf)
        ubx.append(vertical_max)
        ubx.append(horizon_max)
        ubx.append(np.inf)

    # Simulation
    t0 = 0.0
    x0 = np.array([0, 0, 0]).reshape(-1, 1)  # initial state
    x0_ = x0.copy()
    x_m = np.zeros((n_states, N + 1))
    next_states = x_m.copy()
    xs = np.array([1.5, 1.5, 0.0]).reshape(-1, 1)  # final state
    u0 = np.array([0, 0] * N).reshape(-1, 2).T  # np.ones((N, 2)) # controls
    x_c = []  # contains for the history of the state
    u_c = []
    t_c = [t0]  # for the time
    xx = []
    uu = []
    sim_time = 20.0

    # start MPC
    mpciter = 0
    start_time = time.time()
    index_t = []
    # initial test
    while np.linalg.norm(x0 - xs) > 1e-2 and mpciter - sim_time / T < 0.0:
        # set parameter
        c_p = np.concatenate((x0, xs))
        init_control = np.concatenate(
            (u0.T.reshape(-1, 1), next_states.T.reshape(-1, 1))
        )
        t_ = time.time()
        res = solver(x0=init_control, p=c_p, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)

        objective_value = res['f']  
        objective_value_np = objective_value.full().item()

        index_t.append(time.time() - t_)
        estimated_opt = res[
            "x"
        ].full()  # the feedback is in the series [u0, x0, u1, x1, ...]
        u0 = estimated_opt[:N * n_controls].reshape(N, n_controls).T  # (n_controls, N)
        x_m = estimated_opt[N * n_controls:].reshape(N + 1, n_states).T  # [n_states, N]
        x_c.append(x_m.T)
        u_c.append(u0[:, 0])
        t_c.append(t0)
        print("u0 = " , u0)
        print()
        print("x_m = " , x_m)
       

        states = np.zeros((3, u0.shape[1] + 1))
        states[:, 0] = [0 , 0 , 0]
        # Iterative computation of states
        for i in range(N):
            previous = states[: , i]
            B = np.array([[np.cos(previous[2]), 0],
                        [np.sin(previous[2]), 0],
                            [0, 1]]) * T
            
            states[:, i + 1] = states[:, i] + B @ u0[:, i]
            print()
            # print("states[:, i] = " , states[:, i])
            # print("B = " , B)
            # print("u0[:, i] = " , u0[:, i])
            print()
        print("states = " , states)
        print("objective_value_np = " , objective_value_np)
        # exit()

        t0, x0, control , u0, next_states = shift_movement(T, t0, x0, u0, x_m, f)
        x0 = ca.reshape(x0, -1, 1)
        x0 = x0.full()
        xx.append(x0)
        uu.append(control)
        # print("xx = " , xx)
        # print("uu = " , uu)
        # exit()
        mpciter = mpciter + 1
    t_v = np.array(index_t)
    # print(t_v.mean())
    # print((time.time() - start_time) / (mpciter))

    draw_result = Draw_MPC_point_stabilization_v1(
        rob_diam=0.3, init_state=x0_, target_state=xs, robot_states=xx
    )
