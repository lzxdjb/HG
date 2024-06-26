#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <Eigen/Dense>
#include <cusparse.h>
#include <cusolverSp.h>
#include<iostream>
using namespace std;
using Eigen::Matrix;
/*******************/
/* iDivUp FUNCTION */
/*******************/
//extern "C" int iDivUp(int a, int b){ return ((a % b) != 0) ? (a / b + 1) : (a / b); }
__host__ __device__ int iDivUp(int a, int b){ return ((a % b) != 0) ? (a / b + 1) : (a / b); }

/********************/
/* CUDA ERROR CHECK */
/********************/
// --- Credit to http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) { exit(code); }
    }
}

extern "C" void gpuErrchk(cudaError_t ans) { gpuAssert((ans), __FILE__, __LINE__); }

/**************************/
/* CUSOLVE ERROR CHECKING */
/**************************/
static const char *_cusolverGetErrorEnum(cusolverStatus_t error)
{
    switch (error)
    {
    case CUSOLVER_STATUS_SUCCESS:
        return "CUSOLVER_SUCCESS";

    case CUSOLVER_STATUS_NOT_INITIALIZED:
        return "CUSOLVER_STATUS_NOT_INITIALIZED";

    case CUSOLVER_STATUS_ALLOC_FAILED:
        return "CUSOLVER_STATUS_ALLOC_FAILED";

    case CUSOLVER_STATUS_INVALID_VALUE:
        return "CUSOLVER_STATUS_INVALID_VALUE";

    case CUSOLVER_STATUS_ARCH_MISMATCH:
        return "CUSOLVER_STATUS_ARCH_MISMATCH";

    case CUSOLVER_STATUS_EXECUTION_FAILED:
        return "CUSOLVER_STATUS_EXECUTION_FAILED";

    case CUSOLVER_STATUS_INTERNAL_ERROR:
        return "CUSOLVER_STATUS_INTERNAL_ERROR";

    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";

    }

    return "<unknown>";
}

inline void __cusolveSafeCall(cusolverStatus_t err, const char *file, const int line)
{
    if (CUSOLVER_STATUS_SUCCESS != err) {
        fprintf(stderr, "CUSOLVE error in file '%s', line %d, error: %s \nterminating!\n", __FILE__, __LINE__, \
            _cusolverGetErrorEnum(err)); \
            assert(0); \
    }
}

extern "C" void cusolveSafeCall(cusolverStatus_t err) { __cusolveSafeCall(err, __FILE__, __LINE__); }

/***************************/
/* CUSPARSE ERROR CHECKING */
/***************************/
static const char *_cusparseGetErrorEnum(cusparseStatus_t error)
{
    switch (error)
    {

    case CUSPARSE_STATUS_SUCCESS:
        return "CUSPARSE_STATUS_SUCCESS";

    case CUSPARSE_STATUS_NOT_INITIALIZED:
        return "CUSPARSE_STATUS_NOT_INITIALIZED";

    case CUSPARSE_STATUS_ALLOC_FAILED:
        return "CUSPARSE_STATUS_ALLOC_FAILED";

    case CUSPARSE_STATUS_INVALID_VALUE:
        return "CUSPARSE_STATUS_INVALID_VALUE";

    case CUSPARSE_STATUS_ARCH_MISMATCH:
        return "CUSPARSE_STATUS_ARCH_MISMATCH";

    case CUSPARSE_STATUS_MAPPING_ERROR:
        return "CUSPARSE_STATUS_MAPPING_ERROR";

    case CUSPARSE_STATUS_EXECUTION_FAILED:
        return "CUSPARSE_STATUS_EXECUTION_FAILED";

    case CUSPARSE_STATUS_INTERNAL_ERROR:
        return "CUSPARSE_STATUS_INTERNAL_ERROR";

    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";

    case CUSPARSE_STATUS_ZERO_PIVOT:
        return "CUSPARSE_STATUS_ZERO_PIVOT";
    }

    return "<unknown>";
}

inline void __cusparseSafeCall(cusparseStatus_t err, const char *file, const int line)
{
    if (CUSPARSE_STATUS_SUCCESS != err) {
        fprintf(stderr, "CUSPARSE error in file '%s', line %Ndims\Nobjs %s\nerror %Ndims: %s\nterminating!\Nobjs", __FILE__, __LINE__, err, \
            _cusparseGetErrorEnum(err)); \
            cudaDeviceReset(); assert(0); \
    }
}

extern "C" void cusparseSafeCall(cusparseStatus_t err) { __cusparseSafeCall(err, __FILE__, __LINE__); }


void dense_to_csr(double* h_A_dense, int Nrows, int Ncols,
                  int* h_A_RowIndices, int* h_A_ColIndices) {
    
    int nnz = 0; // Number of non-zero elements encountered
    
    // Loop through each row
    for (int i = 0; i < Nrows; ++i) {
        h_A_RowIndices[i] = nnz; // Start of the current row
        
        // Loop through each column in the current row
        for (int j = 0; j < Ncols; ++j) {
            double val = h_A_dense[j * Nrows + i]; // Access element in column-major order

            // std::cout<<"val = " <<val<<endl;
            
            // std::cout<<abs(val - 0.0)<<endl;
            if (abs(val - 0.0) > 1e-10) {
                // std::cout<<"j = "<<j<<endl;
                // std::cout<<"value = "<<val<<endl;
                // Found a non-zero element
                h_A_ColIndices[nnz] = j; // Store column index
                nnz++; // Increment non-zero count
            }
        }
    }
    
    // Last entry in h_A_RowIndices should be nnz
    h_A_RowIndices[Nrows] = nnz;
}
/********/
/* MAIN */
/********/

const int ColumnTestColumn = 0;
// typedef Matrix<double , ColumnTestRow , ColumnTestColumn> Test;
// Test testMatrix;

const int RealRow = 9 ;
const int RealColumn = 9;
typedef Matrix<double , RealRow , RealColumn> Real;
Real realMatrix;

const int RowTestRow = 0;

// typedef Matrix<double , RowTestRow , RowTestColumn> RowTest;
// RowTest rowTest;

typedef Matrix<double , RealRow + RowTestRow , RealColumn + ColumnTestColumn> Final;
Final finalMatrix;
 

void mapMatrixToDoubleArray( Final matrix, double* array) {
    // Map Eigen::Matrix to a raw double array
    Eigen::Map<Final>(array, matrix.rows(), matrix.cols()) = matrix;
}

int main()
{
    // --- Initialize cuSPARSE
    cusparseHandle_t handle;    cusparseSafeCall(cusparseCreate(&handle));

    int Nrows = RealRow;                        // --- Number of rows
    int Ncols = RealColumn;                        // --- Number of columns
    int N = Nrows;

    // --- Host side dense matrix
    double *h_A_dense = (double*)malloc(Nrows*Ncols*sizeof(*h_A_dense));

    // --- Column-major ordering
    h_A_dense[0] = 0.0f; h_A_dense[4] = 4.0f; h_A_dense[8] = 0.0f; h_A_dense[12] = 0.0f;h_A_dense[16] = 1.0f;
    h_A_dense[1] = 0.0f; h_A_dense[5] = 2.0f; h_A_dense[9] = 3.0f; h_A_dense[13] = 0.0f;h_A_dense[17] = 10.0f;
    h_A_dense[2] = 0.0f; h_A_dense[6] = 0.0f; h_A_dense[10] = 3.0f; h_A_dense[14] = 7.0f;h_A_dense[18] = 0.0f;
    h_A_dense[3] = 0.0f; h_A_dense[7] = 0.0f; h_A_dense[11] = 9.0f; h_A_dense[15] = 0.0f;h_A_dense[19] = 0.0f;

    h_A_dense = new double[Nrows * Ncols];
     double values[] = {
       -1.5, 0, 0, 0.5, 0, 0, 0, 0, 0, 
0, -0.1, 0, 0, 0.1, 0, 0, 0, 0, 
0, 0, -15, -4.20735, 2.70151, 5, 0, 0, 0, 
0.5, 0, -4.20735, -4.83229, 1.81859, 4.20735, 0.5, 0, 0, 
0, 0.1, 2.70151, 1.81859, -2.36771, -2.70151, 0, 0.1, 0, 
0, 0, 5, 4.20735, -2.70151, -20, -4.20735, 2.70151, 5, 
0, 0, 0, 0.5, 0, -4.20735, -4.83229, 1.81859, 4.20735, 
0, 0, 0, 0, 0.1, 2.70151, 1.81859, -2.36771, -2.70151, 
0, 0, 0, 0, 0, 5, 4.20735, -2.70151, -20, 
    };

    for(int i = 0; i < Nrows * Ncols; ++i) {
        h_A_dense[i] = values[i];
    }

    // testMatrix.setZero();

    realMatrix = Real::Map(h_A_dense, Nrows, Ncols);
    finalMatrix.bottomRightCorner(RealRow , RealColumn) = realMatrix;

    std::cout<<"finalMatrix = \n"<<finalMatrix<<std::endl;

    h_A_dense = new double[finalMatrix.rows() * finalMatrix.cols()];

    // Map the matrix to the double array
    mapMatrixToDoubleArray(finalMatrix, h_A_dense);

    Nrows = RealRow + RowTestRow;                     // --- Number of rows
    Ncols = RealColumn + ColumnTestColumn;                        // --- Number of columns
    N = Nrows;




    //create device array and copy host to it
    double *d_A_dense;  gpuErrchk(cudaMalloc(&d_A_dense, Nrows * Ncols * sizeof(*d_A_dense)));
    gpuErrchk(cudaMemcpy(d_A_dense, h_A_dense, Nrows * Ncols * sizeof(*d_A_dense), cudaMemcpyHostToDevice));

    // Real *d_A_dense;  gpuErrchk(cudaMalloc(&d_A_dense, Nrows * Ncols * sizeof(*d_A_dense)));
    // gpuErrchk(cudaMemcpy(d_A_dense, h_A_dense, Nrows * Ncols * sizeof(*d_A_dense), cudaMemcpyHostToDevice));


    // --- Descriptor for sparse matrix A
    cusparseMatDescr_t descrA;      cusparseSafeCall(cusparseCreateMatDescr(&descrA));
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    int nnz = 0;                                // --- Number of nonzero elements in dense matrix
    const int lda = Nrows;                      // --- Leading dimension of dense matrix
    // --- Device side number of nonzero elements per row
    int *d_nnzPerVector;    gpuErrchk(cudaMalloc(&d_nnzPerVector, Nrows * sizeof(*d_nnzPerVector)));
    cusparseSafeCall(cusparseDnnz(handle, CUSPARSE_DIRECTION_ROW, Nrows, Ncols, descrA, d_A_dense, lda, d_nnzPerVector, &nnz));
  
    double *d_A;            gpuErrchk(cudaMalloc(&d_A, nnz * sizeof(*d_A)));
    int *d_A_RowIndices;    gpuErrchk(cudaMalloc(&d_A_RowIndices, (Nrows + 1) * sizeof(*d_A_RowIndices)));
    int *d_A_ColIndices;    gpuErrchk(cudaMalloc(&d_A_ColIndices, nnz * sizeof(*d_A_ColIndices)));

    cusparseSafeCall(cusparseDdense2csr(handle, Nrows, Ncols, descrA, d_A_dense, lda, d_nnzPerVector, d_A, d_A_RowIndices, d_A_ColIndices));

    // --- Host side dense matrix
    int *h_A_RowIndices = (int *)malloc((Nrows + 1) * sizeof(*h_A_RowIndices));
    int *h_A_ColIndices = (int *)malloc(nnz * sizeof(*h_A_ColIndices));


    int *Test_h_A_RowIndices = (int *)malloc((Nrows + 1) * sizeof(*h_A_RowIndices));
    int *Test_h_A_ColIndices = (int *)malloc(nnz * sizeof(*h_A_ColIndices));


    dense_to_csr(h_A_dense ,Nrows , Ncols ,  Test_h_A_RowIndices ,Test_h_A_ColIndices);


    for (int i = 0; i < (Nrows + 1); ++i) printf("h_A_RowIndices[%i] = %i \n", i, Test_h_A_RowIndices[i]); printf("\n");

    for (int i = 0; i < nnz; ++i) printf("h_A_ColIndices[%i] = %i \n", i, Test_h_A_ColIndices[i]);


    gpuErrchk(cudaMemcpy(h_A_RowIndices, d_A_RowIndices, (Nrows + 1) * sizeof(*h_A_RowIndices), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_A_ColIndices, d_A_ColIndices, nnz * sizeof(*h_A_ColIndices), cudaMemcpyDeviceToHost));

    std::cout<<"reald " << std::endl;
    for (int i = 0; i < (Nrows + 1); ++i) printf("h_A_RowIndices[%i] = %i \n", i, h_A_RowIndices[i]); printf("\n");

    for (int i = 0; i < nnz; ++i) printf("h_A_ColIndices[%i] = %i \n", i, h_A_ColIndices[i]);

    exit(0);


    // --- Allocating and defining dense host and device data vectors
    double *h_y = (double *)malloc(Nrows * sizeof(double));
    h_y[0] = 100.0;  h_y[1] = 200.0; h_y[2] = 400.0; h_y[3] = 500.0;

    double *d_y;        gpuErrchk(cudaMalloc(&d_y, Nrows * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_y, h_y, Nrows * sizeof(double), cudaMemcpyHostToDevice));

    // --- Allocating the host and device side result vector
    double *h_x = (double *)malloc(Ncols * sizeof(double));
    double *d_x;        gpuErrchk(cudaMalloc(&d_x, Ncols * sizeof(double)));

    // --- CUDA solver initialization
    cusolverSpHandle_t solver_handle;
    cusolverSpCreate(&solver_handle);

    // --- Using LU factorization
    int singularity;
    // cusolveSafeCall(cusolverSpDcsrlsvluHost(solver_handle, N, nnz, descrA, h_A, h_A_RowIndices, h_A_ColIndices, h_y, 0.000001, 0, h_x, &singularity));
    // // --- Using QR factorization
    // //cusolveSafeCall(cusolverSpDcsrlsvqrHost(solver_handle, N, nnz, descrA, h_A, h_A_RowIndices, h_A_ColIndices, h_y, 0.000001, 0, h_x, &singularity));

    // //int rankA;
    // //int *p = (int *)malloc(N * sizeof(int));
    // //double min_norm;
    // //cusolveSafeCall(cusolverSpDcsrlsqvqrHost(solver_handle, N, N, nnz, descrA, h_A, h_A_RowIndices, h_A_ColIndices, h_y, 0.000001, &rankA, h_x, p, &min_norm));

    // printf("Showing the results...\n");
    // for (int i = 0; i < N; i++) printf("%f\n", h_x[i]);

/////

    cusolveSafeCall(cusolverSpDcsrlsvqr(solver_handle, N, nnz, descrA, d_A, d_A_RowIndices, d_A_ColIndices, d_y, 0.000001, 0, d_x, &singularity));

    gpuErrchk(cudaMemcpy(h_x, d_x, Nrows*sizeof(double), cudaMemcpyDeviceToHost));


    printf("Showing the results...\n");

    for (int i = 0; i < N; i++) printf("%f\n", h_x[i]);

}