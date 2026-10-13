# RHI texture shapes on a REAL device (created windowless): 2D with a full
# mip chain, cube maps, 3D volumes, 2D arrays, block-compressed (BC1/BC7) and
# half-float formats. Every mip of every layer is uploaded with a distinct
# byte pattern and read back through Texture::download — pinning subresource
# addressing, row pitches and block packing on both backends. SKIPs (exit 0)
# when no capable GPU exists so the GPU-less CI stays green.
add_executable(Cardinal_Test_GpuTextures main.cpp)
target_link_libraries(Cardinal_Test_GpuTextures PRIVATE
    cardinal::rhi
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_GpuTextures PROPERTIES FOLDER tests)

add_test(NAME gpu_textures COMMAND Cardinal_Test_GpuTextures)
