# SUPERSEDED — see ../2026-10-23-text-ab-identity/

These runs were taken with the stock `calib-L.json` still installed, so the "raw" column was already
f(served) = f_old(raw) and the suite's refit COMPOUNDED the installed transform. The fitted params in
these files are therefore not installable and the SST-5 "calibration makes ECE worse" result was an
artifact of that compounding (T applied twice).

Re-run with the calibrator removed (`jevport-arm.sh <arm> --calib identity`) is in
`../2026-10-23-text-ab-identity/` and is the quotable set.
