import numpy as np

x_m = np.array([[0.00000000e+00, 1.88412886e-01, 4.56426036e-01, 4.56426036e-01],
                [0.00000000e+00, 1.88079096e-37, 3.17207287e-01, 3.17207287e-01],
                [3.52648305e-37, 8.69261870e-01, 8.04872102e-01, 8.04872102e-01]])

u0 = np.array([[9.42064431e-01, 2.07636407e+00, -1.72721296e-38],
               [4.34630935e+00, -3.21948841e-01, 0.00000000e+00]])
Q = np.array([[1.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.1]])
R = np.array([[0.5, 0.0], [0.0, 0.05]])
x_ref = np.array([1.5, 1.5, 0.0])

a = 3

x_selected = x_m[:, :a]
u_selected = u0[:, :a]
print("x_selected = " , x_selected.shape)
print("u_selected = " , u_selected.shape)

x_ref_broadcasted = np.tile(x_ref[:, np.newaxis], (1, a))
print(x_selected.shape)



cost = np.trace( (x_selected -x_ref_broadcasted ).T@ Q @ (x_selected - x_ref_broadcasted ) )+ np.trace(u_selected.T @ R @ u_selected)

print("cost = " , cost)
