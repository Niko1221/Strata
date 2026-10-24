<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="380">
  </picture>
</p>

<p align="center"><a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.de.md">Deutsch</a> · <a href="README.fr.md">Français</a> · <b>Español</b> · <a href="README.pt-BR.md">Português</a></p>

<h3 align="center">Ejecuta un modelo de última generación en tu propio PC gaming</h3>

<p align="center">Compatible con NVIDIA, AMD, Intel · Windows, Linux, Mac · libre y de código abierto</p>

<p align="center">
  <a href="https://github.com/Niko1221/Strata/releases/latest"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?include_prereleases&label=release" alt="Última versión"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="Licencia MIT"></a>
  <a href="https://github.com/Niko1221/Strata/stargazers"><img src="https://img.shields.io/github/stars/Niko1221/Strata?style=flat" alt="Estrellas en GitHub"></a>
  <a href="https://buymeacoffee.com/strataengine"><img src="https://img.shields.io/badge/Buy%20me%20a%20coffee-FFDD00?logo=buymeacoffee&logoColor=black" alt="Invítame a un café"></a>
</p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="Un jardín con una pagoda de vóxeles que escribió el modelo de Strata, funcionando en el navegador"></a><br>
<sub>Un jardín con una pagoda de vóxeles, hecho con un solo prompt en una RTX 5070 con Strata (IQ3_S, contexto de 128K) ·
<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">vídeo completo (49 s)</a></sub></p>

<p align="center">
  <a href="#instalación"><b>Instalación</b></a> ·
  <a href="#modelos">Modelos</a> ·
  <a href="#qué-velocidad-tiene">Velocidad</a> ·
  <a href="#uso">Uso</a> ·
  <a href="#cómo-funciona">Cómo funciona</a> ·
  <a href="#algo-ha-ido-mal">Ayuda</a> ·
  <a href="#apoya-strata">Apoyar</a>
</p>

Strata ejecuta **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)**, un modelo de 125 mil millones
de parámetros que normalmente necesita un servidor, en un PC normal. Desde 0.2.4 también ejecuta **DeepSeek V4.1-Flash**
en la Terminal. Conversa, escribe código, lee imágenes y funciona con tus aplicaciones y tus agentes de programación. Tus
chats y tus archivos se quedan en tu PC.

- **Funciona en un PC gaming.** Con 32 GB entre RAM y memoria gráfica ya puedes empezar.
- **Funciona con tus herramientas.** Claude Code, Codex CLI, OpenCode y otros agentes se conectan por las API de OpenAI y de Anthropic.
- **Lee imágenes**, activado por defecto.
- **Gratis y privado.** Licencia MIT, sin cuenta, sin nube, sin telemetría ([qué descarga y cuándo](docs/SAFETY.md)).

## Lo que necesitas

**Necesitas al menos 32 GB de memoria: RAM y memoria gráfica (VRAM) juntas.** En un Mac o un Ryzen AI Max, es la memoria
unificada. Un PC de 32 GB con una tarjeta de 12 a 16 GB ejecuta los tamaños más pequeños, y con 64 GB funcionan todos los
tamaños.

| Funciona en | Guía |
| --- | --- |
| **NVIDIA** GeForce RTX serie 20, 30, 40 y 50, 12 GB o más | [Instalación](#instalación) |
| **AMD** Radeon RX 6800 y posteriores, Radeon AI PRO R9700 | [Tarjetas AMD](docs/AMD_HIP.md) |
| **Intel** Arc | [Intel Arc](docs/INTEL_ARC.md) |
| **Mac con Apple Silicon**, M1 o posterior, macOS 15 o posterior | [Mac](docs/MAC.md) |
| **AMD Ryzen AI Max (Strix Halo)** y otros PC con memoria unificada | [Strix Halo](docs/STRIX_HALO.md) |
| Tarjetas y procesadores antiguos | [GPU antiguas](docs/OLDER_GPUS.md) |

También necesitas unos 80 GB de disco libre (un SSD), Windows 10 / 11, Linux o macOS, y un controlador gráfico actual. Dos
o tres tarjetas de la misma marca pueden compartir un modelo ([multi-GPU](docs/MULTI_GPU.md)).

## Instalación

### 1. La aplicación Strata (la más fácil)

Descarga el archivo para tu ordenador, ábrelo y pulsa Instalar en el modelo que recomienda.

**Windows 10 / 11**

[![Windows, NVIDIA, controlador 580 o posterior](https://img.shields.io/badge/Windows-NVIDIA%20%C2%B7%20driver%20580%2B-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-windows-cuda13.exe)
[![Windows, NVIDIA, controlador antiguo](https://img.shields.io/badge/Windows-NVIDIA%20%C2%B7%20older%20driver-4C7A00?style=for-the-badge&logo=nvidia&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-windows-cuda12.exe)
[![Windows, AMD RX 9000 o R9700](https://img.shields.io/badge/Windows-AMD%20RX%209000%20%C2%B7%20R9700-ED1C24?style=for-the-badge&logo=amd&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-windows-hip.exe)
[![Windows, AMD RX 7000 o Ryzen AI Max](https://img.shields.io/badge/Windows-AMD%20RX%207000%20%C2%B7%20Ryzen%20AI%20Max-B3131B?style=for-the-badge&logo=amd&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-windows-hip-rdna3.exe)

**Linux**

[![Linux, NVIDIA, controlador 580 o posterior](https://img.shields.io/badge/Linux-NVIDIA%20%C2%B7%20driver%20580%2B-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-nvidia-cuda13.AppImage)
[![Linux, NVIDIA, controlador antiguo](https://img.shields.io/badge/Linux-NVIDIA%20%C2%B7%20older%20driver-4C7A00?style=for-the-badge&logo=nvidia&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-nvidia-cuda12.AppImage)
[![Linux, AMD RX 9000 o R9700](https://img.shields.io/badge/Linux-AMD%20RX%209000%20%C2%B7%20R9700-ED1C24?style=for-the-badge&logo=amd&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-amd-rx9070.AppImage)
[![Linux, AMD RX 7000](https://img.shields.io/badge/Linux-AMD%20RX%207000-B3131B?style=for-the-badge&logo=amd&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-amd-rx7900.AppImage)
[![Linux, AMD Ryzen AI Max](https://img.shields.io/badge/Linux-Ryzen%20AI%20Max-8A0E15?style=for-the-badge&logo=amd&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-amd-strixhalo.AppImage)

**Mac** (Apple Silicon, macOS 15 o posterior)

[![Mac, Apple Silicon](https://img.shields.io/badge/Mac-Apple%20Silicon-333333?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-mac.dmg)

¿Tu controlador NVIDIA es anterior a 580 (`nvidia-smi` lo muestra)? Elige el archivo para controlador antiguo. Una tarjeta
AMD RX 6000 usa el archivo de RX 7000, que busca solo el motor correcto. Todos los archivos están en la
[página de versiones](https://github.com/Niko1221/Strata/releases/latest).

| Primer arranque | Qué hacer |
| --- | --- |
| **Windows** | Ejecuta el `.exe`. No necesita permisos de administrador. Si Windows muestra "Windows protegió tu PC", elige **Más información** y luego **Ejecutar de todas formas**. |
| **Linux** | Haz el archivo ejecutable (`chmod +x`) y lánzalo. Necesita FUSE 2: `sudo apt install libfuse2t64` (Ubuntu 24.04 y posteriores) o `libfuse2` (22.04). |
| **Mac** | Abre el `.dmg` y arrastra Strata a Aplicaciones. |

La descarga del modelo es grande (40-70 GB). Si se detiene, continúa donde se quedó.

### 2. La Terminal (una línea)

El comando `strata`, para scripts, servidores y agentes de IA. Es también la forma de ejecutar DeepSeek.

Linux y Mac:

```sh
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

Windows (PowerShell):

```powershell
irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

Después arráncalo. Te hace una pregunta cada vez (modelo, contexto, imágenes, quién puede conectarse), descarga lo que
necesita y empieza a servir:

```sh
strata launch
```

No necesita permisos de administrador. La instalación comprueba el SHA-256 de todo lo que descarga. En Windows, abre una
ventana de terminal nueva después de la instalación. Todas las pantallas y opciones: [docs/TERMINAL_LAUNCH.md](docs/TERMINAL_LAUNCH.md).

### 3. Que lo instale tu IA

Pega esto en Claude Code, Cursor, Codex, GitHub Copilot u otro asistente de programación:

```text
Set up Strata on this PC for me: https://github.com/Niko1221/Strata - follow docs/AI_SETUP.md in that repository.
```

El asistente revisa tu tarjeta gráfica, tu RAM y tu disco, elige el modelo adecuado, lo instala y lo arranca, y te explica
cómo conectar tus aplicaciones. Las herramientas de IA también pueden instalar, arrancar y detener Strata mediante su
[servidor MCP](docs/MCP_SERVER.md).

¿Prefieres compilarlo tú? Mira [Compilar desde el código fuente](#compilar-desde-el-código-fuente).

## Modelos

La aplicación y `strata launch` recomiendan un tamaño según tu memoria. Los tamaños pequeños son más rápidos. Los grandes
son algo más inteligentes.

### ¿Poca memoria? Prueba Gyro

[Gyro](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF) de agentionai es la forma más pequeña de ejecutar
Qwen3.8-Flash-Next. Funciona en NVIDIA, AMD, Intel Arc y Apple Silicon.

| Tamaño | Memoria que necesita (RAM + VRAM) | Descarga | Para qué sirve |
| --- | ---: | ---: | --- |
| **Gyro-S** | 27,6 GB | 58,5 GB | PC donde no cabe nada más. 9 tokens/s en un Mac mini de 32 GB. |
| **Gyro-M** | 35 GB | 92,0 GB | Calidad parecida a IQ3_XXS con un 20 % menos de memoria de GPU. |

Arráncalo con `strata launch --model gyro-s`. Detalles: [docs/GYRO.md](docs/GYRO.md).

### Los tamaños de Qwen3.8-Flash-Next

| Tu memoria (RAM + VRAM) | Elige | Por qué |
| --- | --- | --- |
| **32 GB** | **Gyro-S**, o el **[Coder](docs/MODELS.md#coder)** para código | El Coder conserva los expertos que usa el código |
| **40 GB** | **[Swift 1.5](docs/MODELS.md#swift-15)** IQ2_XS (39,2 GB) o **Gyro-M** | Swift piensa menos tiempo antes de responder |
| **48 GB** | **IQ3_XXS** (47,0 GB) | Más lento, mejor |
| **64 GB** | **IQ3_S** (54,8 GB) | Iguala al modelo completo en las pruebas publicadas |
| **96 GB o más** | **UD-IQ4_XS** hasta **Q8_0** de Unsloth | Lo más cerca del modelo completo |

Con menos RAM de la que indica una fila también puede funcionar, solo que más lento: Strata pasa solo a su
[modo de poca RAM](docs/MODELS.md#a-big-graphics-card-and-little-ram). También hay:
[NVFP4](docs/NVFP4.md) para tarjetas NVIDIA, [Qwen3.6-35B-A3B](docs/QWEN36.md) (más pequeño, varias peticiones a la vez) y
[el Uncensored IQ3_XXS de OrcaRouter](docs/MODELS.md#orcarouter-uncensored-iq3_xxs). Todos los tamaños y lo que cabe en cada equipo:
[docs/MODELS.md](docs/MODELS.md).

### DeepSeek V4.1-Flash (Terminal, desde 0.2.4)

Un modelo mucho más grande que funciona con una sola tarjeta gráfica de 12 GB y de 46 a 64 GB de RAM. Strata mantiene en
memoria las partes más usadas y lee el resto de tu SSD. [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF)
hizo los conjuntos. Funciona en tarjetas NVIDIA RTX de las series 30 a 50 y en la Radeon AI PRO R9700. Elígelo en
`strata launch`. La aplicación de escritorio todavía no lo ofrece.

| Conjunto | Descarga | Calidad | Escribe código a |
| --- | ---: | --- | --- |
| **IQ1_M** (1,75 bits) | 239 GB | El más pequeño, más débil con prosa sencilla | aún sin medir en un PC |
| **IQ2_XXS** (2.00 bits) | 257 GB | Buen equilibrio | 10 tokens/s (RTX 3060), 5 (RTX 5070), 48 (4x R9700) |
| **IQ2_XS** (2.25 bits) | 274 GB | Lo más cerca del original | 15 tokens/s (una R9700), 80 (4x R9700) |
| **IQ4_NL** (tabla Engram) | incluida en cada descarga de arriba | La tabla de consulta que comparten los tres conjuntos | |

Arráncalo con `strata launch --model ist-das-2.00bpw` (o `ist-das-1.75bpw`, `ist-das-2.25bpw`). Cifras de calidad,
velocidades por PC e instalación manual: [docs/DEEPSEEK41.md](docs/DEEPSEEK41.md).

## ¿Qué velocidad tiene?

Medido con una tarjeta gráfica de 12 GB (RTX 5070, 64 GB de RAM). Resultados con otras tarjetas:
[velocidad de cada tamaño](docs/MODELS.md#how-fast-is-each-size) y [resultados de la comunidad](docs/COMMUNITY_BENCHMARKS.md).

| Modelo | Las respuestas aparecen a | Lee tu prompt a |
| --- | --- | --- |
| **Q2_0** | `████████████` 94 tokens/s | `████████████` 2.650 tokens/s |
| **IQ2_XS** | `██████████░░` 79 tokens/s | `█████████░░░` 2.090 tokens/s |
| **IQ3_XXS** | `████████░░░░` 62 tokens/s | `████████░░░░` 1.750 tokens/s |
| **IQ3_S** | `███████░░░░░` 53 tokens/s | `███████░░░░░` 1.620 tokens/s |
| **Coder** | `███████░░░░░` 55 tokens/s | `██████████░░` 2.180 tokens/s |

Con una tarjeta AMD de 16 GB (RX 9070 XT, 47 GB de RAM):

| Modelo | Las respuestas aparecen a | Lee tu prompt a |
| --- | --- | --- |
| **Q2_0** | `████████░░░░` 60 tokens/s | `█████░░░░░░░` 1.160 tokens/s |
| **IQ2_XS** | `███████░░░░░` 52 tokens/s | `█████░░░░░░░` 1.110 tokens/s |
| **Coder** | `██████░░░░░░` 44 tokens/s | `██████░░░░░░` 1.420 tokens/s |

Un token equivale a unos ¾ de palabra, así que 60 tokens por segundo es más rápido de lo que puedes leer. "Lee tu prompt"
es lo rápido que Strata procesa un documento largo, código o un historial de chat.

## Uso

### En la aplicación

Un chat, un **Monitor** del modelo y de tu GPU, CPU y RAM, una página de **Modelos** y los **Ajustes**. El chat puede
trabajar en una carpeta de tu PC. En modo **Manual** pregunta antes de cambiar un archivo o ejecutar un comando. Puede buscar
en la web (DuckDuckGo o Bing, sin cuenta). Las palabras que buscas son lo único que sale de tu PC.

<p align="center"><img src="docs/media/app-chat.png" width="900" alt="La aplicación Strata: un chat donde el agente construyó un pequeño jardín con una pagoda de vóxeles, con la página abierta en la vista previa a la derecha"><br>
<sub>El agente escribió un jardín con una pagoda de vóxeles en un solo archivo HTML, y la página se abre en la vista previa junto al chat</sub></p>

### En la Terminal

<p align="center"><img src="docs/media/terminal-monitor.png" width="900" alt="La vista general del monitor de la Terminal de Strata mientras el modelo escribe una respuesta"><br>
<sub><code>strata monitor</code> mientras el modelo responde: velocidad, GPU, memoria y las últimas solicitudes</sub></p>

```sh
strata launch                              # guided: model, context, pictures, access, then serve
strata launch --model q2_0 --yes --no-tui  # no questions: serve and print the address
strata launch --status                     # what is being served now
strata launch --stop                       # stop it
strata monitor                             # watch speed, GPU, memory and requests
strata update                              # get the newest engine
strata doctor                              # check the installed files
```

Para llegar a Strata desde tu móvil u otro PC, usa `strata launch --access lan`. Crea una clave de API para ti.
`strata --help` lo lista todo, y [docs/TERMINAL.md](docs/TERMINAL.md) lo explica.

### Con agentes de IA y otras aplicaciones

Un agente instala Strata con el comando de una línea de arriba y ejecuta `strata launch --model NAME --yes --no-tui`. Los
pasos, las opciones y los códigos de salida están en [docs/AI_SETUP.md](docs/AI_SETUP.md). Las reglas del repositorio para agentes: [AGENTS.md](AGENTS.md).

| Tu aplicación habla | URL base |
| --- | --- |
| API de OpenAI (la mayoría de aplicaciones y agentes) | `http://127.0.0.1:8080/v1` (cualquier clave de API, cualquier nombre de modelo) |
| API de Anthropic (Claude Code) | `http://127.0.0.1:8080`, sin `/v1`. Claude Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:8080` |
| API OpenAI Responses (Codex CLI) | `http://127.0.0.1:8080/v1/responses` ([configuración](docs/DETAILS.md#the-responses-api-and-codex-cli)) |

`strata connect` muestra los ajustes exactos para OpenCode, Codex CLI, Claude Code, oh-my-pi, pi, Hermes Agent y OpenClaw,
y `strata connect opencode --write` los escribe por ti (con una copia de seguridad).

### Conviene saber

- **Las imágenes** vienen activadas en todos los modelos que pueden leerlas. Adjunta o pega una imagen en el chat, o envíala desde tu aplicación. Detalles: [docs/IMAGES.md](docs/IMAGES.md).
- **El razonamiento** puede estar desactivado, bajo, medio o alto, en el menú del chat o en "reasoning effort" de tu aplicación. Desactivado es lo más rápido.
- **El primer mensaje** de un chat se lee entero, unos 1 minuto por cada 30.000 tokens. Los mensajes siguientes empiezan en segundos.
- **Una petición a la vez** es lo normal. Para varias, usa `--parallel N` ([BATCHING.md](docs/BATCHING.md)). En una tarjeta de 12 GB, esto hace más lenta cada respuesta.

## Compilar desde el código fuente

<details>
<summary>Windows, Linux y Mac desde el repositorio</summary>

[Descarga Strata](https://github.com/Niko1221/Strata/archive/refs/heads/main.zip) y descomprímelo, o haz `git clone`.
**Windows:** haz doble clic en `START-HERE.bat`. **Linux:** ejecuta `./setup.sh` en la carpeta de Strata.

**Mac (Apple Silicon, macOS 15 o posterior):** instala una vez las Command Line Tools de Apple con `xcode-select --install`
(no hace falta Xcode). Abre la Terminal en la carpeta de Strata y ejecuta `./setup.sh`. El script compila el motor para Mac
en tu Mac (unos 7 minutos), hace las mismas preguntas, descarga el modelo y lo arranca. La próxima vez, `./setup.sh` lo
arranca enseguida. Qué tamaño cabe en tu Mac: [docs/MAC.md](docs/MAC.md).

El instalador detecta tu tarjeta y prepara el motor adecuado. Pregunta qué modelo y qué tamaño quieres, cuánto contexto
(cuánto texto mantiene el modelo en mente) y si debe leer imágenes. Pulsa Enter para aceptar la respuesta recomendada.
Luego descarga el modelo (unos 70 GB) y lo arranca. Si la descarga se detiene, vuelve a ejecutar el script y continúa. Tu
navegador abre la página de Strata en `http://127.0.0.1:8080`.

> Mientras el modelo arranca, tu PC puede ir lento o dejar de responder durante 1-3 minutos (más tiempo la primera vez).
> Strata carga 35-55 GB en tu RAM y bloquea una parte para la tarjeta gráfica. Espera y no cierres la ventana.

La próxima vez, ejecuta de nuevo `START-HERE.bat` (o `./setup.sh`). Arranca enseguida y no descarga nada dos veces. Cierra
su ventana para detener el modelo. `UPDATE.bat` (`./update.sh`) actualiza Strata sin arrancarlo. Actualización, Docker,
varias tarjetas, dónde van los archivos y todas las opciones: [docs/INSTALL.md](docs/INSTALL.md).

Para medir tu PC, ejecuta `BENCH.bat` (`./bench.sh`). Tarda 5-10 minutos y escribe un informe que puedes compartir
([cómo](docs/COMMUNITY_BENCHMARKS.md#run-the-benchmark)).

**Opcional: más velocidad en una tarjeta cuando todo el modelo cabe en la RAM.** Dos ajustes que se activan a mano, de
RubenTorney (#1904, #1905). Ayudan cuando usas una sola tarjeta gráfica y tu RAM guarda todos los expertos, por ejemplo con
Q2_0 o IQ3_XXS en un PC de 64 GB. Abre `strata-<model>.json` en la carpeta de datos de Strata ([dónde](docs/INSTALL.md#where-things-are-stored)) y añade:

- `"--adapt-async", "1"` a la lista `"args"`. Así la caché de expertos se actualiza en un hilo auxiliar en vez de detener la
  respuesta. Medido en una RTX 5070 (Windows, Q2_0): +4-5 % en decodificación, +7-10 % en código y documentos largos. Las
  respuestas pueden diferir un poco de las del ajuste por defecto y son iguales de una ejecución a otra.
- `"STRATA_PIN_BY_NEED": "1"` en el bloque `"env"` (Windows). Windows solo puede bloquear aproximadamente la mitad de la RAM
  para la tarjeta gráfica, y este ajuste bloquea primero los expertos que la tarjeta necesita antes. Medido en la RTX 5070: +4,4 % en lectura del prompt.

```json
"args": [ ..., "--adapt-async", "1" ],
"env": { "STRATA_PIN_BY_NEED": "1" }
```

Reinicia Strata después del cambio. Para deshacerlo, quita las dos entradas.

</details>

## ¿Cómo funciona?

Los modelos como este suelen ejecutarse en servidores con cientos de gigabytes de memoria gráfica. Tu tarjeta gráfica tiene
12-24 GB. Strata hace que el modelo quepa repartiendo el trabajo por todo tu PC. Imagina una biblioteca.

- El modelo es un equipo de 24.576 pequeños especialistas ("expertos"), como 24.576 libros de consulta. Cada palabra necesita solo 10 de ellos.
- **Tu tarjeta gráfica es el escritorio.** Es pequeño y rápido. Los unos 4.000 libros que más se usan están abiertos encima.
- **Tu RAM son las estanterías de la sala de lectura.** Los 24.576 libros están ahí. Cuando una palabra necesita un libro que
  no está en el escritorio, tu procesador lo coge enseguida de la estantería mientras la tarjeta sigue trabajando.
- **Tu SSD es el archivo.** Guarda una tabla de consulta de 29 GB. El modelo lee unas pocas filas por palabra, así que no
  hace falta que esté en la RAM.

<p align="center"><img src="docs/media/how-it-works.svg" width="860" alt="Los 24.576 expertos del modelo: los más usados en la tarjeta gráfica, todos en la RAM, una tabla de consulta en el SSD"></p>

Un pequeño ayudante también adivina las siguientes palabras. El modelo grande las comprueba todas a la vez y se queda con
las correctas. Obtienes la misma respuesta, entre 1,6 y 1,8 veces antes. Los textos largos se leen en trozos grandes, de
hasta 8.192 tokens a la vez, a más de 1.000 tokens por segundo.

<p align="center"><img src="docs/media/guess-and-check.svg" width="860" alt="Un pequeño ayudante adivina las siguientes palabras; el modelo grande las comprueba todas a la vez y se queda con las correctas"></p>

La explicación más larga: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md). Cada parte con sus cifras:
[los detalles](docs/DETAILS.md#how-it-works) y el [artículo](docs/paper/Strata-Paper.pdf).

## ¿Algo ha ido mal?

- **Mi PC se congeló la primera vez que arrancó Strata.** Carga el modelo en la RAM. Espera y no cierres la ventana. ¿Sigue
  congelado después de 10 minutos? Reinicia el PC, cierra otros programas e inténtalo de nuevo, o elige un tamaño más pequeño.
- **Se detuvo durante la descarga o la instalación.** Abre de nuevo la aplicación o `strata launch`. Continúa donde se
  quedó.
- **Va muy lento, o dice "the engine stopped unexpectedly".** Tu PC no tiene bastante RAM libre. Cierra otros programas
  (los navegadores gastan mucha) o elige un tamaño más pequeño, como Q2_0 o IQ2_XS.
- **Dice que no hay motor para este PC.** En la aplicación, abre **Ajustes, Motor** y pulsa **Actualizar motor**. En la
  Terminal, ejecuta `strata update`.
- **Dice que el puerto 8080 ya está en uso.** Strata ya se está ejecutando. Busca su ventana.

Más soluciones: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). ¿Sigues atascado? Abre una
[incidencia](https://github.com/Niko1221/Strata/issues) y adjunta los registros: en la aplicación desde **Ajustes, Motor,
Diagnóstico, Guardar registros**, en la Terminal la salida de `strata doctor`. Para un problema de seguridad, avísanos en
privado: [SECURITY.md](SECURITY.md).

## Créditos y licencia

El modelo es [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), del equipo de Qwen. Lo comprimieron
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), UkisAI (Swift 1.5), agentionai (Gyro)
y Unsloth. ISTA-DASLab hizo también los conjuntos de DeepSeek V4.1-Flash. Strata usa partes de
[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). Todos los créditos: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits).
Strata es de código abierto con la [licencia MIT](LICENSE). Algunas partes y todos los modelos tienen sus propias licencias
([cuáles](docs/HOW_IT_WORKS.md#license)).

## Contribuir

¿Has encontrado un error, o quieres enviar un cambio o un benchmark? Lee primero [CONTRIBUTING.md](CONTRIBUTING.md). Explica
qué incluir para que otros puedan comprobar tu trabajo.

## Apoya Strata

Strata es gratis y tiene licencia MIT. Lo construyo y lo pruebo en el hardware que consigo. Si te funciona bien en tu PC, un café me ayuda
a seguir probando tarjetas nuevas y a arreglar lo que nos cuentas. Sin presión. Usarlo y contar lo que encuentras ayuda igual de mucho.

<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a>

¿Te interesa patrocinar Strata? Escríbeme a través de [GitHub](https://github.com/Niko1221).
