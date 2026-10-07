# strata-metal: the macOS / Apple Silicon engine, Strata's line protocol over llama.cpp's Metal backend
# (docs/MACOS.md).  Entered from the top-level CMakeLists.txt with -DSTRATA_ENABLE_METAL=ON; nothing else there
# is configured.  llama.cpp comes at a pinned commit (STRATA_LLAMA_COMMIT), or from STRATA_LLAMA_DIR, a checkout of
# exactly that commit (offline builds).
set(STRATA_LLAMA_COMMIT "c479922ac520a08969b4c1dc154d7bbb3c386d85")   # 2026-10-06, after qwen4exp (#27742)
set(STRATA_LLAMA_DIR "" CACHE PATH "a llama.cpp checkout at STRATA_LLAMA_COMMIT with metal/patches applied (default: fetched)")

set(GGML_METAL ON CACHE BOOL "" FORCE)
set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)   # the shaders inside the binary: no Xcode needed to run it
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)     # common/speculative: the MTP drafts (--mtp)
set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)       # strata-vision: the image encoder
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)

if(STRATA_LLAMA_DIR)
  add_subdirectory(${STRATA_LLAMA_DIR} llama.cpp EXCLUDE_FROM_ALL)
else()
  # Strata's own Metal kernels (docs/MACOS.md, "For developers"), as patches on the pinned commit; each one is checked
  # by llama.cpp's test-backend-ops and A/B-tested on the model.  STRATA_METAL_PATCHES=OFF builds the plain commit.
  option(STRATA_METAL_PATCHES "apply metal/patches/*.patch to the fetched llama.cpp" ON)
  set(_patch "")
  if(STRATA_METAL_PATCHES)
    file(GLOB _patches "${CMAKE_CURRENT_SOURCE_DIR}/patches/*.patch")
    list(SORT _patches)
    # patch(1), not git apply: the fetched tree lies inside this repository's build folder, and git apply inside
    # another repository resolves the paths against that repository and silently applies nothing
    find_program(STRATA_PATCH patch REQUIRED)
    foreach(_p ${_patches})
      if(_patch)
        list(APPEND _patch COMMAND)
      else()
        set(_patch PATCH_COMMAND)
      endif()
      list(APPEND _patch "${STRATA_PATCH}" -p1 -N -i "${_p}")
    endforeach()
  endif()
  include(FetchContent)
  FetchContent_Declare(llama_cpp URL https://github.com/ggml-org/llama.cpp/archive/${STRATA_LLAMA_COMMIT}.tar.gz ${_patch})
  FetchContent_MakeAvailable(llama_cpp)
endif()

add_executable(strata-metal strata_metal.cpp)
target_compile_features(strata-metal PRIVATE cxx_std_17)
target_compile_definitions(strata-metal PRIVATE STRATA_VERSION="${PROJECT_VERSION}"
                           STRATA_LLAMA_COMMIT="${STRATA_LLAMA_COMMIT}")
target_link_libraries(strata-metal PRIVATE llama llama-common)

# the image encoder (tools/vision): the same source as on a PC, on llama.cpp's mtmd with Metal (--gpu)
add_executable(strata-vision ${CMAKE_CURRENT_SOURCE_DIR}/../tools/vision/strata_vision.cpp)
target_compile_features(strata-vision PRIVATE cxx_std_17)
target_link_libraries(strata-vision PRIVATE mtmd llama)
