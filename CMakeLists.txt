cmake_minimum_required(VERSION 3.13)
project(Degas2-Geom-Debug)

find_package(Omega_h REQUIRED)
add_executable(set_tags src/set_tags.cpp)
target_link_libraries(set_tags PRIVATE Omega_h::omega_h)