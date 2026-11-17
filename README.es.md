# Hakua / Strata

[English](README.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [Français](README.fr.md) · [Polski](README.pl.md) · [Português](README.pt-BR.md) · [简体中文](README.zh-CN.md)

IA local para Windows con chat, tareas de agente, investigación con fuentes y selección de modelos compatibles. Hakua es el fork de zapabob basado en Niko1221/Strata; se mantiene de forma independiente y no es una distribución respaldada por el proyecto original.

## Versión y descargas

<!-- release-061-stable -->
Versión actual: [0.6.1](https://github.com/zapabob/Strata/releases/tag/v0.6.1) (App `0.6.1`, Native Q38 `0.6.1` SM120, Control `0.6.0`). `releaseQualified=false` sigue vigente; la inferencia a ventana completa y la calidad general del modelo no están certificadas.

[0.6.1](https://github.com/zapabob/Strata/releases/tag/v0.6.1) · [Strata-0.6.1-windows-x64-setup.exe](https://github.com/zapabob/Strata/releases/download/v0.6.1/Strata-0.6.1-windows-x64-setup.exe) · [tar.7z](https://github.com/zapabob/Strata/releases/download/v0.6.1/Strata-0.6.1-windows-x64.tar.7z) · [SHA256SUMS](https://github.com/zapabob/Strata/releases/download/v0.6.1/SHA256SUMS.txt)

![Hakua / Strata real-model Japanese chat on Qwen3.8-27B](docs/media/native-qwen38-japanese-chat.png)
*Hakua / Strata Desktop ejecutando `qwen3.8-27b-abliterated-3.69bpw-12GB-MTP.gguf` mediante `strata-q38.exe` (CUDA 13 / RTX 5060 Ti 16 GB).*

## Instalación y actualización

Utilice el instalador NSIS de Windows x64 de este fork. Se instala por usuario e incluye Node, Python y Control privados; no requiere herramientas de desarrollo del sistema. El motor CUDA nativo es un componente separado. El archivo `tar.7z` incluye el instalador, el paquete CUDA, el paquete Control, instrucciones y sumas de verificación, sin pesos de modelos ni datos personales.

Finalice las solicitudes en curso, descargue los modelos, haga una copia de seguridad de los datos y cierre la aplicación incluido el icono de la bandeja. Ejecute el instalador en la carpeta existente para actualizar por sobrescritura conservando modelos, chats, memoria, herramientas, claves y permisos. Importe el componente CUDA comprimido desde *Ya tengo el archivo*, elija un modelo compatible o un GGUF local y cárguelo. Consulte [instalación y reversión](docs/HAKUA_INSTALL.md).

## Novedades de 0.6.1 y funciones del fork

0.6.1 mejora el [ejecutor local GGUF denso para Qwen3.8-27B (`strata-q38.exe`)](docs/QWEN38_DENSE_NATIVE.md) en Windows CUDA 13 / Blackwell (SM120):
- **Polimorfismo de cuantización por tensor (`StrataQuantTensor`)**: admite 11 tipos GGML/Strata (`F32`, `F16`, `BF16`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ3_S`, `IQ2_S`, `IQ4_XS` y PrismML `PQ2_0=142`) por tensor.
- **Prefill por lotes con Tensor Core + KV por lotes con `spin8_triality.cuh` (`KvCache::enqueue_batch`)**: sustituye el GEMV secuencial por `cublasGemmEx` con doble búfer (`prefill_gemm=cublas_tensor_op`, más `STRATA_FP8_PREFILL=1` opcional). En la prueba de 140 tokens de prefill + 8 de decodificación (`--kv spin8_k8v4 --ctx-size 4096`) en RTX 5060 Ti SM120, el prefill bajó de `36.760,7 ms` (`3,81 tok/s`, `--conservative-sync`) a `11.387,4 ms` (`12,29 tok/s`, **3,23x más rápido**).
- **Verificación real MTP `K=3` (`blk.64.nextn.*`)**: compatible con `--spec-draft-n-max 0..3`, `KvCache::rewind` y reversión de estado GDN.
- **Caché KV y perfil japonés**: modos `F16`, `Spin8 K8V4` y `F16-K / Turbo2/3/4-V` (orden `auto`: `f16` → `f16_turbo4` → `spin8_k8v4`), además de perfil opcional de salida en japonés.

`/btw` inicia un subagente real sin herramientas para preguntas paralelas con control de parada propio. `/research` y `/deepresearch` recopilan fuentes con RustXNG Full integrado (las búsquedas envían consultas a proveedores externos configurados). El aprendizaje y la aplicación automática de habilidades de flujo son opciones separadas y están desactivadas por defecto en perfiles nuevos. Consulte [habilidades](docs/WORKFLOW_SKILLS.md) y [búsqueda](docs/BROWSER_SEARCH.md).

## Integración oficial y compatibilidad

La base oficial fijada es [c775728069](https://github.com/Niko1221/Strata/commit/c77572806996349b49995e4f98d41b5294d36a21), correspondiente a 0.2.15 / motor 0.1.60. Las versiones del fork son independientes. La compatibilidad con DeepSeek se mantiene como extensión y bloquea la carga si falta el runtime dedicado. Este paquete de Windows no implica entrada de imagen nativa, motor DeepSeek nativo distribuido ni inferencia certificada en Linux/Mac.

## Límites de protección y mediciones

Conserve los mínimos de 3 GiB de RAM disponible, 4 GiB de margen de commit, al menos 768 MiB de reserva de GPU y límites de disco. 32 GB de RAM y 16 GB de VRAM no garantizan que cualquier modelo o contexto quepa. `releaseQualified=false` sigue vigente y el instalador carece de firma Authenticode de Windows. Las mediciones históricas permanecen en [evidencias de lanzamiento](docs/HAKUA_RELEASE.md).

## Documentación y desarrollo

[Instalación](docs/HAKUA_INSTALL.md) · [Evidencias](docs/HAKUA_RELEASE.md) · [API y hardware](docs/AI_REFERENCE.md) · [Terminal](docs/TERMINAL.md) · [Compilación](docs/SOURCE_INSTALL.md) · [Empaquetado](packaging/README.md) · [Contribuir](CONTRIBUTING.md) · [Seguridad](SECURITY.md) · [Licencia](LICENSE)

```sh
python -m pytest tests/terminal
```

```sh
cd apps/host
pnpm test
pnpm run build
```

### Niko1221/Strata — referencia oficial de Terminal

```sh
curl -fsSL --connect-timeout 15 --max-time 120 https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

```powershell
irm -TimeoutSec 120 https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

[Niko1221/Strata](https://github.com/Niko1221/Strata) · [Hermes Agent](https://github.com/NousResearch/hermes-agent) · [Avisos de terceros](third_party/)
