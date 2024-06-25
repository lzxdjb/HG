#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <Eigen/Dense>

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> SharedMatrix;

// Example dimensions
const int horizon = 2;
const int StateShape = 3;
const int rows = horizon * StateShape;
const int cols = horizon * StateShape;
const int size = rows * cols;

// Allocate memory for the dense matrix on the device
float* d_dense_matrix;
cudaMalloc(&d_dense_matrix, size * sizeof(float));

// Allocate memory for the sparse matrix components on the device
float* d_values;
int* d_colIndices;
int* d_rowPointers;
cudaMalloc(&d_values, size * sizeof(float));
cudaMalloc(&d_colIndices, size * sizeof(int));
cudaMalloc(&d_rowPointers, (rows + 1) * sizeof(int));

// Allocate memory for the solution vector and RHS vector
float* d_x; // Solution vector
float* d_b; // Right-hand side vector
cudaMalloc(&d_x, rows * sizeof(float));
cudaMalloc(&d_b, rows * sizeof(float));


__global__ void denseToCSR(SharedMatrix* dense, float* values, int* colIndices, int* rowPointers, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        int idx = rowPointers[row];
        for (int col = 0; col < cols; ++col) {
            float val = dense[row * cols + col];
            if (val != 0) {
                values[idx] = val;
                colIndices[idx] = col;
                idx++;
            }
        }
        rowPointers[row + 1] = idx;
    }
}



int main() {
    // Define and initialize the dense matrix on the host
    SharedMatrix h_dense_matrix(rows, cols);
    h_dense_matrix << 1, 0, 3, 0, 0, 0,
                      7, 0, 0, 9, 0, 0,
                      0, 0, 0, 0, 5, 0,
                      0, 0, 0, 0, 0, 6;

    // Copy the dense matrix to the device
    cudaMemcpy(d_dense_matrix, h_dense_matrix.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize row pointers
    std::vector<int> h_rowPointers(rows + 1, 0);
    cudaMemcpy(d_rowPointers, h_rowPointers.data(), (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel to convert the dense matrix to CSR format
    int blockSize = 256;
    int numBlocks = (rows + blockSize - 1) / blockSize;
    denseToCSR<<<numBlocks, blockSize>>>(d_dense_matrix, d_values, d_colIndices, d_rowPointers, rows, cols);
    cudaDeviceSynchronize();

    // Calculate the number of non-zero elements
    cudaMemcpy(h_rowPointers.data(), d_rowPointers, (rows + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    int nnz = h_rowPointers[rows];

    // Initialize cuSPARSE
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    // Define the matrix descriptor
    cusparseMatDescr_t descr;
    cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    // Define a right-hand side vector (example)
    std::vector<float> h_b(rows, 1.0f); // Example RHS vector with all ones
    cudaMemcpy(d_b, h_b.data(), rows * sizeof(float), cudaMemcpyHostToDevice);

    // Perform sparse matrix-vector multiplication: d_dense_matrix * d_x = d_b
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, rows, cols, nnz, &alpha, descr,
                   d_values, d_rowPointers, d_colIndices, d_b, &beta, d_x);

    // Copy the result back to the host
    std::vector<float> h_x(rows);
    cudaMemcpy(h_x.data(), d_x, rows * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    cusparseDestroyMatDescr(descr);
    cusparseDestroy(handle);
    cudaFree(d_dense_matrix);
    cudaFree(d_values);
    cudaFree(d_colIndices);
    cudaFree(d_rowPointers);
    cudaFree(d_x);
    cudaFree(d_b);

    // Print the solution
    std::cout << "Solution vector:" << std::endl;
    for (const auto& val : h_x) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
