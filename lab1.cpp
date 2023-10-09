#include <cstdlib>
#include <memory>
#include <iostream>
#include <chrono>

#include <thread>
#include <vector>
#include <mutex>
#include "num_threads.h"
#include "scalability.h"

using namespace std;

double average(const double* V, size_t n)
{
    double acc = 0;
    for (size_t i = 0; i < n; ++i)
        acc += V[i];
    return acc / n;
}


double average_parallel(const double* V, size_t n)
{
    vector<thread> workers;
    double acc = 0.0;
    std::mutex mtx;
    size_t T = get_num_threads();
    //lambda
    auto worker_proc = [V, n, T, &mtx, &acc](size_t t) {
        double myacc = 0;
        for (size_t i = t; i < n; i += T)
            myacc += V[i];
        
        scoped_lock lock(mtx);
        acc += myacc;
    };
    for (size_t t = 1; t <= T - 1; ++t)
        workers.emplace_back(worker_proc, t);
    worker_proc(0);
    for (auto& th:workers) //for (size_t t = 0; t < T - 1; ++t)
        th.join();
	
    return acc / n;
}

int main(int argc, char* argv[])
{
    size_t N = 100000000;
    auto W = make_unique<double[]>(N); //allocate buffer
    for (size_t i = 0; i < N; i++)
        W[i] = i+1;
	
    cout << "Threads,Speedups\n";
    auto scal1 = get_speedups(average, W.get(), N);
    cout << "scal1:" << endl;
    to_csv(scal1);

    auto scal2 = get_speedups(average_parallel, W.get(), N);
    
    cout << "scal2:" << endl;
    to_csv(scal2);
    
    return 0;
}
