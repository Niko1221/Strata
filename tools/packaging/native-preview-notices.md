# Third-party notices for the native preview

Strata: Copyright (c) 2026 Niko1221 and the Strata contributors, MIT.
Source: https://github.com/zapabob/Strata . See licenses/Strata-MIT.txt.

Embedded ggml: Copyright (c) 2023-2026 The ggml authors, MIT.
See licenses/ggml-MIT.txt, copied from the exact dependency checkout used by
the build. The SHA-256 implementation derives from Igor Pavlov's public-domain
implementation based on Wei Dai's Crypto++; see licenses/SHA256.txt.

NVIDIA cuBLAS and cuBLASLt: unmodified cublas64_13.dll and cublasLt64_13.dll
from CUDA Toolkit 13.3. Windows version-resource values are recorded literally
in the manifest; they are not inferred from the Toolkit folder name.
The complete SDK EULA and third-party notices are supplied unmodified as
licenses/NVIDIA-EULA.txt and licenses/NVIDIA-LICENSE.txt. Attachment A lists
cuBLAS and cuBLASLt including versioned filename variants for redistribution
with applications, subject to the agreement. The CUDA runtime is linked into
the engine; its applicable terms are in the same EULA. No NVIDIA driver is
redistributed. The user's installed driver supplies nvcuda.dll.
https://docs.nvidia.com/cuda/eula/index.html

Microsoft Visual C++ runtime: unmodified x64 MSVCP140.dll,
MSVCP140_ATOMIC_WAIT.dll, VCRUNTIME140.dll and VCRUNTIME140_1.dll from the
Visual Studio 2022 VC/Redist Microsoft.VC143.CRT directory. File version is
14.44.35211.0. These release runtimes are not Microsoft preview software.
The build PC has Visual Studio Community 2022 and Build Tools installed.
Community's individual/open-source permissions and section 4 distributable
code terms apply with the Visual Studio 2022 distributable list. The original
vendor Community, Build Tools and runtime licence documents and the installed
Build Tools third-party notices accompany this bundle. Redistribution is part
of Strata, subject to those terms, not a grant under MIT or a standalone runtime
redistribution. Windows Universal CRT and System32 libraries are not bundled.
https://learn.microsoft.com/en-us/visualstudio/releases/2022/redistribution
https://visualstudio.microsoft.com/license-terms/vs2022-ga-community/
https://visualstudio.microsoft.com/license-terms/vs2022-ga-diagnosticbuildtools/
https://visualstudio.microsoft.com/license-terms/vs2022-cruntime/

The original Microsoft licence documents were obtained from the vendor pages
above on 24 October 2026. Their hashes, and every runtime hash, are in the
manifest. No third-party project, model provider or paper author endorses this
preview. No model weights, tokenizer, GUI assets or research training
environment are distributed in the native engine archive.
