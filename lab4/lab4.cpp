
#include <complex>
#include <vector>
#include <cmath>
#include <iostream>
#include <omp.h>

std::vector<std::complex<double>> FFT_Cooley_Tukey_OpenMP(std::vector<std::complex<double>>& x) {
    unsigned long N = x.size();
    int k;
    std::vector<std::complex<double>> X(N);
    if (N == 1) {
        X[0] = x[0];
        return X;
    }
    std::vector<std::complex<double>> d(N/2), e(N/2);
    for (k = 0; k < N/2; k++) {
        e[k] = x[2*k];
        d[k] = x[2*k + 1];
    }
    std::vector<std::complex<double>> D = FFT_Cooley_Tukey_OpenMP(d);
    std::vector<std::complex<double>> E = FFT_Cooley_Tukey_OpenMP(e);

#pragma omp parallel for
    for(k = 0; k < N/2; k++) {
        D[k] *= std::polar(1.0, -2.0 * 3.1415926535897932384 * k/N);
    }
#pragma omp parallel for
    for(k = 0; k < N/2; k++) {
        X[k] = E[k] + D[k];
        X[k + N/2] = E[k] - D[k];
    }
    return X;
}

std::vector<std::complex<double>> FFT_Cooley_Tukey(std::vector<std::complex<double>>& x) {
    unsigned long N = x.size();
    int k;
    std::vector<std::complex<double>> X(N);
    if (N == 1) {
        X[0] = x[0];
        return X;
    }
    std::vector<std::complex<double>> d(N/2), e(N/2);
    for (k = 0; k < N/2; k++) {
        e[k] = x[2*k];
        d[k] = x[2*k + 1];
    }
    std::vector<std::complex<double>> D = FFT_Cooley_Tukey(d);
    std::vector<std::complex<double>> E = FFT_Cooley_Tukey(e);
    
    for(k = 0; k < N/2; k++) {
        D[k] *= std::polar(1.0, -2.0 * 3.1415926535897932384 * k/N);
    }
    for(k = 0; k < N/2; k++) {
        X[k] = E[k] + D[k];
        X[k + N/2] = E[k] - D[k];
    }
    return X;
}

void test_FFT() {
    // Test 1: FFT of a single 1
    std::vector<std::complex<double>> x1(1, 1);
    auto y1 = FFT_Cooley_Tukey_OpenMP(x1);
    std::cout << "Test 1: " << ((y1[0] == std::complex<double>(1, 0)) ? "PASS" : "FAIL") << std::endl;

    // Test 2: FFT of [1, 1, 1, 1]
    std::vector<std::complex<double>> x2(4, 1);
    auto y2 = FFT_Cooley_Tukey_OpenMP(x2);
    std::cout << "Test 2: " << ((y2[0] == std::complex<double>(4, 0)) ? "PASS" : "FAIL") << std::endl;

    // Test 3: FFT of [0, 1, 0, 1]
    std::vector<std::complex<double>> x3 = {0, 1, 0, 1};
    auto y3 = FFT_Cooley_Tukey_OpenMP(x3);
    std::cout << "Test 3: " << ((y3[0] == std::complex<double>(2, 0)) ? "PASS" : "FAIL") << std::endl;
}

int main(int argc, char* argv[])
{
    test_FFT();
    return 0;
}