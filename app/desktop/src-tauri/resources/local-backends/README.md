# Local llama-server runtimes

Stage qualified PrismML and TurboQuant packages in prism/ and turboquant/. Each contains llama-server.exe, its own dependency libraries, licences and a runtime.json source/payload manifest. Model weights stay outside this folder.

The explicit GUI backend choice selects one complete package. A missing selected package is an error. The default Automatic choice keeps the existing generic-GGUF selection.

PrismML enables PQ2_0 model files. TurboQuant cache choices require verified runtime capabilities and compatible attention geometry. Model loading and cache compatibility are tested separately.

See docs/HAKUA_RELEASE.md for the pending qualification and source revisions.
