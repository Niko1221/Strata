& "D:\Workstation\Strata\.venv\Scripts\python.exe" tools\opt\bench_decode.py `
  --start-server "D:\Workstation\Strata\.venv\Scripts\python.exe serve\server.py --engine strata --config strata-iq3_xxs.json --port 8080" `
  --arm-label probe-199k --prompt-tokens 49152 --max-tokens 32 --repeats 1 --seed 1234 `
  --out-json tools\opt\results\probe-199k.json --gpu-lock
