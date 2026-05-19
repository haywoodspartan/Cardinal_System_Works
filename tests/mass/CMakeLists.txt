# Deterministic Mass-ECS regression suite. cardinal::mass is the
# data-oriented entity store (archetype chunks, registration-order
# component bits, swap-remove storage) that crowd/swarm/projectile
# gameplay iterates thousands of agents through. A storage regression
# silently corrupts every Mass system — lost shared components across an
# archetype move, mis-filtered queries, leaked/duplicated entity slots.
# Pure CPU, single-threaded, fully deterministic. Exit 0 = all pass.
add_executable(Cardinal_Test_Mass main.cpp)
target_link_libraries(Cardinal_Test_Mass PRIVATE
    cardinal::mass
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Mass PROPERTIES FOLDER tests)

add_test(NAME mass COMMAND Cardinal_Test_Mass)
