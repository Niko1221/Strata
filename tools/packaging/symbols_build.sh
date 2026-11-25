# tools/packaging/symbols_build.sh - keep the debug symbols of a release build (sourced, POSIX sh).  Owner decision 2026-11-25 (0.2.22):
# every engine package of a release keeps a symbol file, stored OUTSIDE the package and never uploaded (docs/RELEASING.md, "Symbols").
#
#   . "$repo/tools/packaging/symbols_build.sh"
#   ... cmake -G Ninja ... $sym_cmake ...           # -DSTRATA_KEEP_SYMBOLS=ON: host debug information, the machine code does not change
#   symbols_keep elf <repo> <package name> <stage binary>...      # on the STAGE copy, before package_engine.py hashes it
#   symbols_keep macho <repo> <package name> <stage binary>...    # macOS: dSYM, then strip -S
#
# The files land in ${SYMBOLS_DIR:-<repo>/symbols}/<package name>/ (symbols.json lists build id, sha256 and the symbol file of each binary);
# move that folder to the owner PC's C:\r0222\symbols\<package name>\ (release checklist).  STRATA_KEEP_SYMBOLS=0 builds without
# (a test build); a release build that finds no debug information in a binary fails (--require).
sym_cmake="-DSTRATA_KEEP_SYMBOLS=ON"
[ "${STRATA_KEEP_SYMBOLS:-1}" != 0 ] || sym_cmake=""

symbols_keep() {
  [ -n "$sym_cmake" ] || return 0
  _sk_kind=$1; _sk_repo=$2; _sk_pkg=$3
  shift 3
  "${PYTHON:-python3}" "$_sk_repo/tools/packaging/keep_symbols.py" "$_sk_kind" --out "${SYMBOLS_DIR:-$_sk_repo/symbols}/$_sk_pkg" --require "$@"
}
