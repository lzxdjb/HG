#include <Eigen/Dense>
#include <iostream>

typedef double tinytype; // Example type, replace with actual type
const int StateShape = 4; // Example value, replace with actual value

// Define the SparseUpperDualCache matrix type based on your typedef
typedef Eigen::Matrix<tinytype, StateShape * StateShape * 3, 1> SparseUpperDualCache;

void copyToPointer(const SparseUpperDualCache& matrix, tinytype* ptr, int a, int n, int I, int j) {
    std::memcpy(ptr + I, matrix.data() + a, n * sizeof(tinytype));

    // a is the start point of matrix ; n is the copied array length ; I is the start point of ptr
}

int main() {
    // Example dimensions and indices
    int a = 3;
    int n = 1;
    int I = 1;
    int j = 5;

    // Example SparseUpperDualCache matrix initialization
    SparseUpperDualCache matrix;
    matrix.setRandom(); // Random initialization, replace with actual matrix data
    std::cout<<"matrix = \n"<<matrix.transpose()<<std::endl;

    // Example pointer
    tinytype* ptr = new tinytype[j]; // Assuming ptr points to allocated memory

    // Copy operation
    copyToPointer(matrix, ptr, a, n, I, j);

    // Printing ptr to verify
    std::cout << "Elements in ptr:" << std::endl;
    for (int i = 0; i < j; ++i) {
        std::cout << ptr[i] << std::endl;
    }

    delete[] ptr; // Freeing allocated memory

    return 0;
}
