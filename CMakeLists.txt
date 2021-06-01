cmake_minimum_required(VERSION 3.10)
project(enzyme_ex)

set(CMAKE_CXX_STANDARD 20)
set(google_benchmark_path /usr/local CACHE PATH "Path to where Google Benchmark is installed")
set(trilinos_path /home/michael/programming/trilinos CACHE PATH "Path to the Trilinos source")

add_executable(enzyme_ex main.cpp)

link_directories("${google_benchmark_path}/lib")
set(CMAKE_CXX_FLAGS_RELEASE "-march=native -fstrict-aliasing -O3")
target_link_libraries(enzyme_ex benchmark pthread)
target_include_directories(enzyme_ex PUBLIC ${trilinos_path}/packages/sacado/src ${trilinos_path}/packages/sacado/src/mpl)