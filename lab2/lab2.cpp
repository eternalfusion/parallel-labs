#include <iostream>
#include <immintrin.h>
#include <random>
#pragma comment(linker, "/STACK:2000000")
#pragma comment(linker, "/HEAP:2000000")

#include "scalability.h"

using namespace std;


// M <- A+B
//n mod 16 = 0
// F64^(16*16)=F64^16^16 - вектор из 16 векторов-столбцов
void matrix_add(double* M, const double* A, const double* B, size_t n)
{
    for (size_t c = 0; c < n; c++)
    {
        for (size_t r = 0; r < n; r++)
        {
            M[c * n + r] = A[c * n + r] + B[c * n + r];
        }
    }
};

void matrix_avx256(double* M, const double* A, const double* B, size_t n)
{
    size_t N = sizeof(double) * n / sizeof(__m256d);
    for (size_t c = 0; c < n; c += N)
    {
        for (size_t r = 0; r < n; r++)
        {
            __m256d a = _mm256_load_pd(&A[c * n + r]);
            __m256d b = _mm256_load_pd(&B[c * n + r]);
            __m256d z = _mm256_add_pd(a, b);
            _mm256_store_pd(&M[c * n + r], z);
        }
    }
};

void matrix_multiply_avx256(double* M, double* A, double* B, size_t n) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            __m256d sum = _mm256_setzero_pd();
            for (size_t k = 0; k < n; k += 4) {
                __m256d a = _mm256_load_pd(&A[i * n + k]);
                __m256d b = _mm256_load_pd(&B[k * n + j]);
                sum = _mm256_add_pd(sum, _mm256_mul_pd(a, b));
            }
            _mm256_store_pd(&M[i * n + j], sum);
        }
    }
}

void matrix_multiply(double* M, double* A, double* B, size_t n) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            M[i * n + j] = sum;
        }
    }
}

int main(int argc, char* argv[])
{
    size_t N = 1024; // 1048575 when 1048576 max
    double* m = new double[N * N];
    double* a = new double[N * N];
    double* b = new double[N * N];
    
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist6(1,6); // distribution in range [1, 6]

    for (size_t i = 0; i < N*N; i++)
    {
        a[i] = dist6(rng);
        b[i] = dist6(rng);
    }

    auto time1 = get_time(matrix_avx256, m, a, b, N);
    std::cout << "Add straightforward time: " << time1 << std::endl;
    
    auto time3 = get_time(matrix_add, m, a, b, N);
    std::cout << "Add parallel time: " << time3 << std::endl;
    
    auto time2 = get_time(matrix_multiply_avx256, m, a, b, N);
    std::cout << "Parallel multiply time: " << time2 << std::endl;

    auto time4 = get_time(matrix_multiply, m, a, b, N);
    std::cout << "Straightforward multiply time: " << time4 << std::endl;
    
    return 0;
}