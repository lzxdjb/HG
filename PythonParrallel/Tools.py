import torch

class Tools():
    def lu_no_pivoting(A):
        n = A.size(0)
        L = torch.eye(n, dtype=A.dtype, device=A.device)
        U = A.clone()
    
        for i in range(n):
            if U[i, i] == 0:
                raise ValueError("Zero pivot encountered.")
        
            for j in range(i+1, n):
                L[j, i] = U[j, i] / U[i, i]
                U[j, i:] = U[j, i:] - L[j, i] * U[i, i:]

        return L, U
    
    def check_below_diagonal(matrix):
        num_rows = len(matrix)
        num_cols = len(matrix[0]) if num_rows > 0 else 0
    
        for i in range(num_rows):
            for j in range(num_cols):
                if i > j and matrix[i][j] >= 1e-12:
                    print("position: i = ", i , "j = " , j)
                    print("value = " , matrix[i][j])
                    return False
        return True
    
    def compare_matrices(matrix1, matrix2):
        if matrix1.shape != matrix2.shape:
            raise ValueError("Matrices must have the same dimensions for comparison.")
    
        diff = matrix1 != matrix2
        if diff.any():
            print("Matrices are not equal. Differences found at:")
            diff_indices = diff.nonzero(as_tuple=True)
            for i in range(len(diff_indices[0])):
                row = diff_indices[0][i].item()
                col = diff_indices[1][i].item()
                print(f"Position ({row}, {col}): Matrix1 = {matrix1[row, col]}, Matrix2 = {matrix2[row, col]}")
        else:
            print("Matrices are equal.")

    def LUDebug(final_matrix):
        L, U = Tools.lu_no_pivoting(final_matrix)

        # print("right L = " , L[: , :self.horizon * (self.state_shape + self.control_shape)])

        # print("right R = " , U[: , :self.horizon * (self.state_shape + self.control_shape)])

        # print("final_matrix = " , self.final_matrix)


        # Tools.compare_matrices(U[: , :self.horizon * (self.state_shape + self.control_shape)] ,self.final_matrix[: , :self.horizon * (self.state_shape + self.control_shape)] )

        # Tools.compare_matrices(L[: , :self.horizon * (self.state_shape + self.control_shape)] , self.Lower[: , :self.horizon * (self.state_shape + self.control_shape)] )


    
