import torch

def inverse_tridiagonal(matrix):
    n = 4
    
    a = torch.diag(matrix)  # main diagonal
    b = torch.diag(matrix, 1)  # upper diagonal
    c = torch.diag(matrix, -1)  # lower diagonal

    # Step 1: Compute theta values
    theta = torch.zeros(n + 1)

    theta[0] = 1; theta[1] = a[1]
    for i in range(2, n + 1):
        theta[i] = a[i] * theta[i - 1] - b[i - 1] * c[i - 1] * theta[i - 2]

    # Step 2: Compute phi values
    phi = torch.zeros(n + 2)
    phi[n + 1] = 1
    phi[n] = a[n]

    for i in range(n-1, 0, -1):
        phi[i] = a[i] * phi[i + 1] - b[i] * c[i] * phi[i + 2]

    # Step 3: Construct the inverse matrix
    T_inv = torch.zeros(n + 1, n + 1)

    for i in range(1 , n + 1):
        for j in range(1 , n + 1):
            if i < j:
                product_b = torch.prod(b[i:j])
                T_inv[i, j] = ((-1) ** (i + j)) * product_b * theta[i - 1] * phi[j + 1] / theta[-1]
            elif i == j:
                T_inv[i, j] = theta[i - 1] * phi[j + 1] / theta[-1]
            else:  # i > j
                product_c = torch.prod(c[j:i])
                T_inv[i, j] = ((-1) ** (i + j)) * product_c * theta[j - 1] * phi[i + 1] / theta[-1]

    return T_inv

# Example usage
horizon = 3
stateshape = 3
n = horizon * stateshape

# Create a random tridiagonal matrix for testing
# For simplicity, let's assume horizon * stateshape = 4 for example
matrix = torch.tensor([[1.0, 2.0, 0.0, 0.0],
                       [3.0, 4.0, 5.0, 0.0],
                       [0.0, 6.0, 7.0, 8.0],
                       [0.0, 0.0, 9.0, 10.0]])


# Inserting a row of zeros
row_of_zeros = torch.zeros(1, matrix.size(1))
jjmatrix = torch.cat((row_of_zeros, matrix), dim=0)

# Inserting a column of zeros at the beginning
column_of_zeros = torch.zeros(jjmatrix.size(0), 1)
jjmatrix = torch.cat((column_of_zeros, jjmatrix), dim=1)

inverse_matrix = inverse_tridiagonal(jjmatrix)
inverse_matrix = inverse_matrix[1:, :]
inverse_matrix = inverse_matrix[:, 1:]
print(inverse_matrix)

print("real one = " , matrix.inverse())
