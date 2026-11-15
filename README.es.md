# Hakua / Strata

[English](README.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [Français](README.fr.md) · [Polski](README.pl.md) · [Português](README.pt-BR.md) · [简体中文](README.zh-CN.md)

IA local en Windows para conversar, trabajar con agentes e investigar con fuentes. Hakua es la bifurcación de Niko1221/Strata mantenida de forma independiente por zapabob; no es una distribución avalada por el proyecto original.

## Versión y descargas

<!-- release-060-candidate -->
Este código es el candidato de integración 0.6.0. La aplicación, Terminal, Control y el motor nativo comparten la versión de código 0.6.0. Siguen pendientes las comprobaciones de compilación, instalación y publicación; el número de versión no demuestra que haya archivos disponibles. Se conserva la [versión 0.5.9](https://github.com/zapabob/Strata/releases/tag/v0.5.9) verificada. Consulte el registro de entrega para conocer el estado real de 0.6.0.

[0.6.0](https://github.com/zapabob/Strata/releases/tag/v0.6.0) · [Strata-0.6.0-windows-x64-setup.exe](https://github.com/zapabob/Strata/releases/download/v0.6.0/Strata-0.6.0-windows-x64-setup.exe) · [tar.7z](https://github.com/zapabob/Strata/releases/download/v0.6.0/Strata-0.6.0-windows-x64.tar.7z) · [SHA256SUMS](https://github.com/zapabob/Strata/releases/download/v0.6.0/SHA256SUMS.txt)

## Instalar y actualizar

Utilice el instalador NSIS para Windows x64 de esta bifurcación. La instalación es por usuario e incluye Node, Python y Control privados, sin herramientas de desarrollo del sistema. El motor CUDA firmado es un componente aparte. El archivo tar.7z debe incluir el mismo instalador, componentes, instrucciones y sumas de comprobación, sin pesos de modelos ni datos personales. Extraiga primero 7z y después tar; compruebe BUNDLE-SHA256SUMS.txt.

Termine las solicitudes, descargue los modelos de memoria, guarde una copia de los datos y cierre la aplicación, incluida la bandeja. Instale sobre la carpeta existente. Conserve modelos, conversaciones, memoria, herramientas, claves y autorizaciones. Importe el motor comprimido con la opción de archivo existente y cargue un modelo compatible o GGUF local. El nombre no demuestra compatibilidad con el motor. [Instalación y reversión](docs/HAKUA_INSTALL.md).

## Funciones conservadas

`/btw` inicia un subagente real sin herramientas para preguntas paralelas, con su propio control de parada. `/research` y `/deepresearch` reúnen fuentes. La búsqueda web puede usar RustXNG Full incluido. Las consultas llegan a los proveedores externos habilitados; la búsqueda no es completamente local. El número de fuentes no garantiza exactitud.

El aprendizaje de habilidades y su aplicación automática se activan por separado; ambos están desactivados en perfiles nuevos. Se conservan preferencias, habilidades privadas editables, revisiones y deshacer. También se mantienen las operaciones de archivos, el historial y memoria delimitados, las rutas compatibles PrismML/TurboQuant, la memoria de expertos limitada, la integridad de packs y los diagnósticos. [Habilidades](docs/WORKFLOW_SKILLS.md) · [Búsqueda](docs/BROWSER_SEARCH.md).

## Integración oficial y compatibilidad

La referencia oficial es [09b6038869](https://github.com/Niko1221/Strata/commit/09b603886925994d0d4461857b01321ef3aa431c), de la generación 0.2.14 / motor 0.1.60: red, selección de borradores, procesamiento de prompts, memoria y correcciones de plataforma. Las versiones de la bifurcación son independientes. Las rutas DeepSeek retiradas se conservan como extensión; la ausencia de un ejecutable dedicado debe impedir la carga. Este paquete Windows no acredita entrada nativa de imágenes, distribución de un motor nativo DeepSeek ni inferencia validada en Linux/Mac.

## Protección y límites medidos

Se mantienen 3 GiB de RAM disponible, 4 GiB de margen de memoria comprometida, al menos 768 MiB reservados en GPU y los mínimos de disco. La protección de RAM de Windows observa carga y generación. 32 GB de RAM y 16 GB de VRAM no garantizan cualquier modelo o contexto. La inferencia de ventana completa, la calidad general y el uso simultáneo de GPU siguen sin calificación: releaseQualified=false. El instalador no tiene firma Windows Authenticode; las firmas de componentes y del feed son independientes. Las mediciones históricas mantienen sus versiones en el [registro](docs/HAKUA_RELEASE.md).

## Documentación y desarrollo

[Install](docs/HAKUA_INSTALL.md) · [Release evidence](docs/HAKUA_RELEASE.md) · [API and hardware](docs/AI_REFERENCE.md) · [Terminal](docs/TERMINAL.md) · [Source build](docs/SOURCE_INSTALL.md) · [Packaging](packaging/README.md) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [License](LICENSE)

```sh
python -m pytest tests/terminal
```

```sh
cd apps/host
npm test
npm run build
```

### Niko1221/Strata — upstream Terminal reference

```sh
curl -fsSL --connect-timeout 15 --max-time 120 https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

```powershell
irm -TimeoutSec 120 https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

[Niko1221/Strata](https://github.com/Niko1221/Strata) · [Hermes Agent](https://github.com/NousResearch/hermes-agent) · [Fuentes y avisos de terceros](third_party/)
