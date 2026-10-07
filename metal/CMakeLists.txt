# strata-metal: the macOS / Apple Silicon engine, Strata's line protocol over llama.cpp's Metal backend
# (docs/MACOS_PLAN.md).  Entered from the top-level CMakeLists.txt with -DSTRATA_ENABLE_METAL=ON; nothing else there
# is configured.  llama.cpp comes at a pinned commit (STRATA_LLAMA_COMMIT), or from STRATA_LLAMA_DIR, a checkout of
# exactly that commit (offline builds).
set(STRATA_LLAMA_COMMIT "c479922ac520a08969b4c1dc154d7bbb3c386d85")   # 2026-10-06, after qwen4exp (#27742)
set(STRATA_LLAMA_DIR "" CACHE PATH "a llama.cpp checkout at STRATA_LLAMA_COMMIT (default: fetched)")

set(GGML_METAL ON CACHE BOOL "" FORCE)
set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)   # the shaders inside the binary: no Xcode needed to run it
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)

if(STRATA_LLAMA_DIR)
  add_subdirectory(${STRATA_LLAMA_DIR} llama.cpp EXCLUDE_FROM_ALL)
else()
  include(FetchContent)
  FetchContent_Declare(llama_cpp URL https://github.com/ggml-org/llama.cpp/archive/${STRATA_LLAMA_COMMIT}.tar.gz)
  FetchContent_MakeAvailable(llama_cpp)
endif()

add_executable(strata-metal strata_metal.cpp)
target_compile_features(strata-metal PRIVATE cxx_std_17)
target_compile_definitions(strata-metal PRIVATE STRATA_VERSION="${PROJECT_VERSION}"
                           STRATA_LLAMA_COMMIT="${STRATA_LLAMA_COMMIT}")
target_link_libraries(strata-metal PRIVATE llama)
