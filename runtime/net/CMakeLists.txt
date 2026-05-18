add_library(cardinal_net STATIC
    src/net.cpp
    src/net_udp.cpp
    src/net_replication.cpp
    include/cardinal/net/net.hpp
    include/cardinal/net/replication.hpp)
add_library(cardinal::net ALIAS cardinal_net)

target_include_directories(cardinal_net PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

# scene added for the replication layer (RepState carries scene::Vec3).
# Transport core stays scene-free; only replication.* pulls it.
target_link_libraries(cardinal_net
    PUBLIC  cardinal::core cardinal::scene cardinal::compile_flags)

# UDP backend uses Winsock 2 (socket / bind / recvfrom / WSAStartup).
# PUBLIC: Transport::create_udp() is part of the public surface, so any
# target that calls it (samples, tests, the eventual MMO host) pulls in
# net_udp.obj and must link ws2_32 too. PRIVATE here silently broke that
# the moment a consumer actually used UDP.
if(WIN32)
    target_link_libraries(cardinal_net PUBLIC ws2_32)
endif()

set_target_properties(cardinal_net PROPERTIES FOLDER runtime)
