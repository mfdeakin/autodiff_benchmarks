#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include <Sacado_No_Kokkos.hpp>

#include <benchmark/benchmark.h>

#if 0
extern double __enzyme_autodiff(void *, double);
#define enzyme_autodiff __enzyme_autodiff
#else
double enzyme_autodiff(void *, double) {
  return std::numeric_limits<double>::quiet_NaN();
}
#endif

template <typename Real = double>
Real f(Real x) {
  return pow(x, 4) + exp(-x * x) + 16.0 + sin(x) + sinh(x) / cosh(x);
}

double f_dx(double x) {
  return 4.0 * pow(x, 3) - 2.0 * x * exp(-x * x) + cos(x) +
         (pow(cosh(x), 2) - pow(sinh(x), 2)) / pow(cosh(x), 2);
}

std::vector<double> rand_x() {
  std::random_device rd;
  std::ranlux48_base gen(rd());
  std::uniform_real_distribution dist(-128.0, 128.0);
  std::vector<double> x(128);
  for (size_t i = 0; i < x.capacity(); ++i) {
    x[i] = dist(gen);
  }
  return x;
}

static void BM_Null(benchmark::State &state) {
  auto x = rand_x();
  size_t i = 0;
  while (state.KeepRunning()) {
    x[i];
    ++i;
    i %= x.capacity();
  }
}
BENCHMARK(BM_Null);

static void BM_ExplicitDiff(benchmark::State &state) {
  auto x = rand_x();
  size_t i = 0;
  while (state.KeepRunning()) {
    benchmark::DoNotOptimize(f_dx(x[i]));
    ++i;
    i %= x.capacity();
  }
}
BENCHMARK(BM_ExplicitDiff);

static void BM_EnzymeDiff(benchmark::State &state) {
  auto x = rand_x();
  size_t i = 0;
  while (state.KeepRunning()) {
    benchmark::DoNotOptimize(enzyme_autodiff(reinterpret_cast<void *>(f<double>), x[i]));
    ++i;
    i %= x.capacity();
  }
}
BENCHMARK(BM_EnzymeDiff);

static void BM_SacadoDiff(benchmark::State &state) {
  auto x = rand_x();
  size_t i = 0;
  using Real = Sacado::Fad::SFad<double, 2>;
  while (state.KeepRunning()) {
    Real x_ad = x[i];
    x_ad.diff(0, 1);
    benchmark::DoNotOptimize(f(x_ad));
    ++i;
    i %= x.capacity();
  }
}
BENCHMARK(BM_SacadoDiff);

BENCHMARK_MAIN();
