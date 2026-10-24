"""Check Muse's CMake build graph without pretending to compile other backends.

The temporary projects use the real repository CMake files. Only unavailable
toolkit discovery, the SYCL subproject and unrelated outer targets are replaced.
Run with: python -m unittest tools.muse.test_build
"""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
CMAKE = shutil.which("cmake")


@unittest.skipUnless(CMAKE, "CMake is needed to check the build graph")
class BuildPolicyTests(unittest.TestCase):
    def configure(self, directory: Path, body: str) -> subprocess.CompletedProcess[str]:
        # Forward slashes work in CMake paths on Windows as well as Unix.
        root = ROOT.as_posix()
        (directory / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.24)\n"
            "project(muse_build_policy LANGUAGES CXX)\n"
            f'set(STRATA_TEST_ROOT [==[{root}]==])\n'
            + body,
            encoding="utf-8",
        )
        return subprocess.run(
            [CMAKE, "-S", str(directory), "-B", str(directory / "build")],
            capture_output=True, text=True, timeout=60,
        )

    def assert_configured(self, result: subprocess.CompletedProcess[str]) -> None:
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_gfx906_keeps_muse_cpu_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.configure(Path(temporary), r'''
add_library(ggml-base INTERFACE)
add_library(strata_artifact INTERFACE)
# The outer non-Windows build emits no manifest; its real helper is exercised
# separately below, through the root CMake project.
function(strata_utf8_manifest target)
endfunction()
# The outer gfx906 backend supplies these HIP aliases, not NVIDIA libraries.
add_library(CUDA::cudart INTERFACE IMPORTED)
add_library(CUDA::cublas INTERFACE IMPORTED)
function(find_package package)
  if(package STREQUAL "CUDAToolkit")
    set_property(GLOBAL PROPERTY MUSE_REQUESTED_NVIDIA_TOOLKIT TRUE)
    return()
  endif()
  _find_package(${ARGV})
endfunction()
set(STRATA_ENABLE_CUDA ON)
set(STRATA_HIP_GFX906 ON)
set(STRATA_BUILD_TESTS ON)
add_subdirectory("${STRATA_TEST_ROOT}/src/muse" muse)
get_property(requested GLOBAL PROPERTY MUSE_REQUESTED_NVIDIA_TOOLKIT)
if(requested)
  message(FATAL_ERROR "Muse requested an NVIDIA toolkit for the HIP gfx906 backend")
endif()
foreach(target IN ITEMS strata_muse strata-muse muse_model_test)
  if(NOT TARGET ${target})
    message(FATAL_ERROR "Missing CPU Muse target: ${target}")
  endif()
endforeach()
get_target_property(sources strata_muse SOURCES)
get_target_property(definitions strata_muse COMPILE_DEFINITIONS)
get_target_property(links strata_muse LINK_LIBRARIES)
foreach(value IN LISTS sources definitions links)
  if(value MATCHES "cuda_matvec|STRATA_MUSE_CUDA|CUDA::")
    message(FATAL_ERROR "NVIDIA adapter leaked into HIP Muse: ${value}")
  endif()
endforeach()
''')
            self.assert_configured(result)

    def test_sycl_muse_is_rejected_before_entering_sycl(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            result = self.configure(directory, r'''
set(STRATA_ENABLE_SYCL ON CACHE BOOL "" FORCE)
set(STRATA_ENABLE_MUSE ON CACHE BOOL "" FORCE)
set(STRATA_ENABLE_CUDA OFF CACHE BOOL "" FORCE)
set(STRATA_ENABLE_HIP OFF CACHE BOOL "" FORCE)
# Reaching this stub proves the unsupported request was not rejected early.
function(add_subdirectory directory)
  if(directory STREQUAL "sycl")
    file(WRITE "${CMAKE_BINARY_DIR}/entered-sycl" "entered")
    return()
  endif()
  _add_subdirectory(${ARGV})
endfunction()
_add_subdirectory("${STRATA_TEST_ROOT}" strata)
''')
            output = result.stdout + result.stderr
            self.assertFalse((directory / "build" / "entered-sycl").exists(), output)
            self.assertNotEqual(result.returncode, 0, output)
            self.assertIn("muse", output.lower())
            self.assertIn("sycl", output.lower())

    def test_windows_manifest_is_attached_to_both_executables(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.configure(Path(temporary), r'''
# Avoid fetching GGML: only its target's presence matters to this graph check.
add_library(ggml-base INTERFACE)
set(STRATA_NATIVE_EXPERTS OFF CACHE BOOL "" FORCE)
set(STRATA_ENABLE_MUSE ON CACHE BOOL "" FORCE)
set(STRATA_BUILD_TESTS ON CACHE BOOL "" FORCE)
set(STRATA_ENABLE_SYCL OFF CACHE BOOL "" FORCE)
set(STRATA_ENABLE_CUDA OFF CACHE BOOL "" FORCE)
set(STRATA_ENABLE_HIP OFF CACHE BOOL "" FORCE)
set(STRATA_HIP_GFX906 OFF CACHE BOOL "" FORCE)
# Exercise the real shared manifest helper. Emulate MSVC only inside Muse so
# Unix hosts do not need a Windows compiler or receive unrelated MSVC flags.
function(add_subdirectory directory)
  if(directory STREQUAL "src/muse")
    set(MSVC TRUE)
  endif()
  _add_subdirectory(${ARGV})
endfunction()
_add_subdirectory("${STRATA_TEST_ROOT}" strata)
foreach(target IN ITEMS strata-muse muse_model_test)
  get_target_property(sources ${target} SOURCES)
  list(FIND sources "${STRATA_TEST_ROOT}/cmake/strata-utf8.manifest" manifest)
  if(manifest EQUAL -1)
    message(FATAL_ERROR "Shared Windows UTF-8 manifest missing from ${target}")
  endif()
endforeach()
''')
            self.assert_configured(result)

    def test_missing_row_converter_dependency_has_clear_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.configure(Path(temporary), r'''
set(STRATA_NATIVE_EXPERTS OFF CACHE BOOL "" FORCE)
set(STRATA_ENABLE_MUSE ON CACHE BOOL "" FORCE)
set(STRATA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(STRATA_ENABLE_SYCL OFF CACHE BOOL "" FORCE)
set(STRATA_ENABLE_CUDA OFF CACHE BOOL "" FORCE)
set(STRATA_ENABLE_HIP OFF CACHE BOOL "" FORCE)
set(STRATA_HIP_GFX906 OFF CACHE BOOL "" FORCE)
add_subdirectory("${STRATA_TEST_ROOT}" strata)
''')
            output = result.stdout + result.stderr
            self.assertNotEqual(result.returncode, 0, output)
            self.assertIn("STRATA_NATIVE_EXPERTS=ON", output)
            self.assertIn("Muse", output)


if __name__ == "__main__":
    unittest.main()
