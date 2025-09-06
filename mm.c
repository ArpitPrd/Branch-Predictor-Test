#define N 100
/**
 * @brief multiplies in the block
 */
void multiply_block(int n, double *A, double *B, double *C, int block_size, int i, int j, int k) {
    for (int ii = i; ii < i + block_size && ii < n; ii++) {
        for (int jj = j; jj < j + block_size && jj < n; jj++) {
            for (int kk = k; kk < k + block_size && kk < n; kk++) {
                C[ii * n + jj] += A[ii * n + kk] * B[kk * n + jj];
            }
        }
    }
}

/**
 * @brief for initializing a matrix
 */
void initialize_matrix_to_zero(double* C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0.0;
        }
    }
}

/**
 * @brief multiplies across the matrices
 */
void mat_mul_blocked(int n, double *A, double *B, double *C, int block_size) {
    
    initialize_matrix_to_zero(C, n);
    
    for (int i = 0; i < n; i += block_size) {
        for (int j = 0; j < n; j += block_size) {
            for (int k = 0; k < n; k += block_size) {
                multiply_block(n, A, B, C, block_size, i, j, k);
            }
        }
    }
}

int main() {
    double A[N][N], B[N][N], C[N][N];    
    mat_mul_blocked(N, (double *)A, (double *)B, (double *)C, 8);
}