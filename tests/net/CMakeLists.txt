# Deterministic networking regression suite. Locks the five net arcs:
# quantization round-trip, snapshot seq-gating under reorder, client
# interpolation (lerp + clamp), loopback loss-sim determinism + its
# zero/max-loss invariants, and reliable lifecycle exactness under
# total Unreliable loss. cardinal::net brings core+scene transitively;
# compile_flags applies the same /W4 /WX (or -Werror) as the engine,
# so the test is held to the engine's own bar.
add_executable(Cardinal_Test_Net main.cpp)
target_link_libraries(Cardinal_Test_Net PRIVATE
    cardinal::net
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Net PROPERTIES FOLDER tests)

add_test(NAME net COMMAND Cardinal_Test_Net)

# Real-socket integration: stands up a server + client UDP transport on
# 127.0.0.1 and drives the *same* Replicator over them — proving the
# "swap the factory ⇒ MMO" thesis end to end. Platform-agnostic at the
# source level (uses only the Transport abstraction); skips clean where
# create_udp() returns null. ws2_32 arrives transitively now that it's
# PUBLIC on cardinal::net.
add_executable(Cardinal_Test_NetUdp main_udp.cpp)
target_link_libraries(Cardinal_Test_NetUdp PRIVATE
    cardinal::net
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_NetUdp PROPERTIES FOLDER tests)

add_test(NAME net_udp COMMAND Cardinal_Test_NetUdp)

# Reliability under real injected loss: the cumulative-ack / retransmit
# / reorder / dedupe path, the one netcode machinery nothing else
# exercises. Wall-clock bounded (UDP retransmit is wall-timed); same
# skip-clean contract.
add_executable(Cardinal_Test_NetUdpLoss main_udp_loss.cpp)
target_link_libraries(Cardinal_Test_NetUdpLoss PRIVATE
    cardinal::net
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_NetUdpLoss PROPERTIES FOLDER tests)

add_test(NAME net_udp_loss COMMAND Cardinal_Test_NetUdpLoss)
