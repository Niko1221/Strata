# Hakua / Strata

[English](README.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [Français](README.fr.md) · [Polski](README.pl.md) · [Português](README.pt-BR.md) · [简体中文](README.zh-CN.md)

IA local para Windows con chat, tareas de agente, investigación con fuentes y selección de modelos compatibles. Hakua es el fork de zapabob basado en Niko1221/Strata; se mantiene de forma independiente y no es una distribución respaldada por el proyecto original.

## Qué cambia en 0.6.7

Esta versión fusiona la semántica de Strata oficial 0.2.16 conservando todas las funciones del fork (no se aplicaron las eliminaciones oficiales). El motor CUDA nativo de Qwen3.8 está ahora conectado a la API oficial: `strata.exe` informa `qwen35` en `--families` y sirve el GGUF real de Qwen3.8-27B mediante el protocolo estándar `--serve`. Compilado y verificado con CUDA 13.5 / SM120. El rendimiento de decodificación está a la par de Prism llama.cpp (ambos usan la misma clase de kernel DP4A); una comparación A/B inversa y caliente mostró que agrupar las escrituras F16 de KV reduce el coste de KV en el prefill un 27 %, manteniendo logits idénticos byte a byte. Se incluyen la compresión de RAM previa al arranque y el aviso del límite de commit de Sync-Strata para 32 GB. La generación de ventana completa y la calidad general del modelo siguen sin verificarse (`releaseQualified=false`).

## Versión y descargas

<!-- release-062-stable -->
Versión actual de la app: [0.6.7](https://github.com/zapabob/Strata/releases/tag/v0.6.7) (App `0.6.7`; Terminal, Control y motor nativo `0.6.3`). Esta versión solo para Windows no es GitHub Latest de forma intencionada; Latest sigue siendo `0.6.3`. El motor GPU sigue siendo el componente firmado por separado [`v0.6.4-preview.1`](https://github.com/zapabob/Strata/releases/tag/v0.6.4-preview.1). `releaseQualified=false` sigue vigente; la inferencia a ventana completa y la calidad general del modelo no están certificadas.

[0.6.7](https://github.com/zapabob/Strata/releases/tag/v0.6.7) · [EXE de instalación](https://github.com/zapabob/Strata/releases/download/v0.6.7/Strata-0.6.7-windows-x64-setup.exe) · [tar.7z](https://github.com/zapabob/Strata/releases/download/v0.6.7/Strata-0.6.7-windows-x64.tar.7z) · [SHA256SUMS](https://github.com/zapabob/Strata/releases/download/v0.6.7/SHA256SUMS.txt) · [BUNDLE-SHA256SUMS](https://github.com/zapabob/Strata/releases/download/v0.6.7/BUNDLE-SHA256SUMS.txt)

![Hakua / Strata real-model Japanese chat on Qwen3.8-27B](docs/media/native-qwen38-japanese-chat.png)
*Hakua / Strata Desktop ejecutando `qwen3.8-27b-abliterated-3.69bpw-12GB-MTP.gguf` mediante `strata-q38.exe` (CUDA 13 / RTX 5060 Ti 16 GB).*

## GGUF recomendado para el prototipo Qwen3.8

Para probar el chat de texto local en una RTX 5060 Ti con 16 GB de VRAM, recomendamos el [GGUF de cuantización mixta de 3,69 bpw de soyaakinohara](https://huggingface.co/soyaakinohara/qwen3.8-27b-abliterated-3.69bpw-12GB-MTP.gguf) que usamos en las pruebas. Gracias a **soyaakinohara** por cuantizarlo y publicarlo, y a **AEON-7** por los pesos de origen. Descarga el modelo por separado y empieza con un contexto de 4096 tokens. El nombre del archivo no garantiza que funcione con 12 GB de VRAM.

La [versión preliminar nativa firmada para CUDA 13.5](https://github.com/zapabob/Strata/releases/tag/v0.6.4-preview.1) superó dos turnos en japonés, la interrupción de una respuesta y la descarga del modelo de memoria en la aplicación 0.6.3. En dos pruebas de primera solicitud redujo un 35–43 % el tiempo de prefill F16 corto frente a la ruta nativa anterior; Prism (llama.cpp) siguió decodificando más rápido. [Configuración, clave pública y límites](docs/Q38_NATIVE_PREFILL_PREVIEW.md).

## Instalación y actualización

Utilice el instalador NSIS de Windows x64 de este fork. Se instala por usuario e incluye Node, Python y Control privados; no requiere herramientas de desarrollo del sistema. El motor CUDA nativo es un componente separado. El archivo `tar.7z` incluye el instalador, el paquete CUDA, el paquete Control, instrucciones y sumas de verificación, sin pesos de modelos ni datos personales.

Finalice las solicitudes en curso, descargue los modelos, haga una copia de seguridad de los datos y cierre la aplicación incluido el icono de la bandeja. Ejecute el instalador en la carpeta existente para actualizar por sobrescritura conservando modelos, chats, memoria, herramientas, claves y permisos. Importe el componente CUDA comprimido desde *Ya tengo el archivo*, elija un modelo compatible o un GGUF local y cárguelo. Consulte [instalación y reversión](docs/HAKUA_INSTALL.md).

## Novedades de 0.6.2 y funciones del fork

0.6.2 mejora el [ejecutor local GGUF denso para Qwen3.8-27B (`strata-q38.exe`)](docs/QWEN38_DENSE_NATIVE.md) en Windows CUDA 13 / Blackwell (SM120):
- **Polimorfismo de cuantización por tensor (`StrataQuantTensor`)**: admite 11 tipos GGML/Strata (`F32`, `F16`, `BF16`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ3_S`, `IQ2_S`, `IQ4_XS` y PrismML `PQ2_0=142`) por tensor.
- **Prefill por lotes Tensor Core + GDN multi-token + Atención Spin8 fusionada + Decode con CUDA Graph (CUDA 13.5 SM120)**:
  - Prefill de 140 tokens (`--kv spin8_k8v4 --ctx-size 4096`): de `36.760,7 ms` (`3,81 tok/s`, `--conservative-sync`) a **`1.558,3 ms` (`89,8 tok/s` en caliente, 23,6x más rápido)**.
  - Decode estable (`GEN 32`, 31 pasos, GEMV empaquetado + CUDA Graph con barreras cada 8 capas): de `206,4 ms/paso` a **`85,13 ms/paso` (`11,75 pasos/s`, 2,42x más rápido)**.
  - **Medición de contexto largo (`--kv spin8_k8v4`, RTX 5060 Ti 16 GB)**: **65K (`65.536`)** superado (prefill de 1.024 tokens `10.813,5–11.220,7 ms`, `91,3–94,7 tok/s`; decode `670,4 ms` / 7 pasos); **96K (`98.304`)** superado (`required_with_reserve=15,09 GB`, prefill de 140 tokens `1.968,7 ms`); **131K (`131.072`)** rechazado por la reserva de 768 MiB (`required=15,97 GB > available=15,91 GB`).
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

## Cambios en 0.6.3

- El prefill nativo de Q38 usa GEMM de Tensor Core de cuBLAS con activaciones FP16 escaladas por token y un tile de pesos persistente en L2 de 20 MiB. El decode usa GEMV dp4a Q8_1 alineado según el formato y un CUDA Graph con LayerFence(8).
- La admisión de contexto largo elige el tamaño de chunk de prefill mediante una escalera de VRAM medida. La reserva de GPU de 768 MiB y los mínimos de 3 GiB de RAM y 4 GiB de commit no cambian.
- Los fixtures numéricos normalizan los pesos empaquetados sintéticos a amplitudes reales sin relajar las tolerancias estrictas de `_exact_q8`. El GEMV no alineado recurre a un kernel escalar de carga por bytes.
- Verificación: 17/17 pruebas nativas superadas; compute-sanitizer informa 0 errores y 0 riesgos en memcheck/synccheck/racecheck. La calidad general del modelo, la inferencia de ventana completa y la concurrencia de GPU siguen sin cualificar (`releaseQualified=false`).
- Base de investigación: notas de lanzamiento de NVIDIA CUDA 13.5 (corrección del algoritmo 66 de cuBLASLt con Tensor Memory) y el informe de arquitectura Qwen3.8-Next (híbrido GDN/atención global 3:1, QSA con compresión r=4 y presupuesto K=2048, residual con compuerta, embedding de n-gramas precargado desde el host).
