<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="380">
  </picture>
</p>

<p align="center"><a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.de.md">Deutsch</a> · <a href="README.fr.md">Français</a> · <a href="README.es.md">Español</a> · <b>Português</b></p>

<h3 align="center">Rode um modelo de ponta no seu próprio PC gamer</h3>

<p align="center">Suporte a NVIDIA, AMD, Intel · Windows, Linux, Mac · gratuito e de código aberto</p>

<p align="center">
  <a href="https://github.com/Niko1221/Strata/releases/latest"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?include_prereleases&label=release" alt="Última versão"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="Licença MIT"></a>
  <a href="https://github.com/Niko1221/Strata/stargazers"><img src="https://img.shields.io/github/stars/Niko1221/Strata?style=flat" alt="Estrelas no GitHub"></a>
  <a href="https://buymeacoffee.com/strataengine"><img src="https://img.shields.io/badge/Buy%20me%20a%20coffee-FFDD00?logo=buymeacoffee&logoColor=black" alt="Me pague um café"></a>
</p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="Um jardim de pagodes em voxels que o modelo do Strata escreveu, rodando no navegador"></a><br>
<sub>Um jardim de pagodes em voxels, gerado com um único prompt, rodando numa RTX 5070 com o Strata (IQ3_S, contexto de 128K) ·
<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">vídeo completo (49 s)</a></sub></p>

<p align="center">
  <a href="#instalação"><b>Instalação</b></a> ·
  <a href="#modelos">Modelos</a> ·
  <a href="#qual-é-a-velocidade">Velocidade</a> ·
  <a href="#como-usar">Como usar</a> ·
  <a href="#como-funciona">Como funciona</a> ·
  <a href="#algo-deu-errado">Ajuda</a> ·
  <a href="#apoie-o-strata">Apoiar</a>
</p>

O Strata roda o **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)**, um modelo de 125 bilhões de parâmetros que
normalmente precisa de um servidor, num PC comum. A partir da 0.2.4 ele também roda o **DeepSeek V4.1-Flash** no Terminal. Ele
conversa, escreve código, lê imagens e funciona com os seus aplicativos e agentes de programação. Suas conversas e arquivos
ficam no seu PC.

- **Roda num PC gamer.** 32 GB somando memória RAM e memória da placa de vídeo já bastam para começar.
- **Funciona com as suas ferramentas.** Claude Code, Codex CLI, OpenCode e outros agentes se conectam pelas APIs da OpenAI e da Anthropic.
- **Lê imagens**, ativado por padrão.
- **Gratuito e privado.** Licença MIT, sem conta, sem nuvem, sem telemetria ([o que ele baixa e quando](docs/SAFETY.md)).

## O que você precisa

**Você precisa de pelo menos 32 GB de memória: RAM e memória da placa de vídeo (VRAM) somadas.** Num Mac ou num Ryzen AI Max, isso é a
memória unificada. Um PC de 32 GB com uma placa de 12 a 16 GB roda os tamanhos menores, e 64 GB rodam todos os tamanhos.

| Funciona em | Guia |
| --- | --- |
| **NVIDIA** GeForce RTX série 20, 30, 40 e 50, 12 GB ou mais | [Instalação](#instalação) |
| **AMD** Radeon RX 6800 e mais novas, Radeon AI PRO R9700 | [Placas AMD](docs/AMD_HIP.md) |
| **Intel** Arc | [Intel Arc](docs/INTEL_ARC.md) |
| **Mac com Apple Silicon**, M1 ou mais novo, macOS 15 ou mais novo | [Mac](docs/MAC.md) |
| **AMD Ryzen AI Max (Strix Halo)** e outros PCs de memória unificada | [Strix Halo](docs/STRIX_HALO.md) |
| Placas e processadores mais antigos | [GPUs antigas](docs/OLDER_GPUS.md) |

Você também precisa de cerca de 80 GB de espaço livre no disco (um SSD), Windows 10 / 11, Linux ou macOS, e um driver de vídeo
atualizado. Duas ou três placas da mesma marca podem dividir um modelo ([multi-GPU](docs/MULTI_GPU.md)).

## Instalação

### 1. O app Strata (o mais fácil)

Baixe o arquivo para o seu computador, abra e clique em instalar no modelo que ele recomenda.

**Windows 10 / 11**

[![Windows, NVIDIA, driver 580 ou mais novo](https://img.shields.io/badge/Windows-NVIDIA%20%C2%B7%20driver%20580%2B-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-windows-cuda13.exe)
[![Windows, NVIDIA, driver mais antigo](https://img.shields.io/badge/Windows-NVIDIA%20%C2%B7%20older%20driver-4C7A00?style=for-the-badge&logo=nvidia&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-windows-cuda12.exe)
[![Windows, AMD RX 9000 ou R9700](https://img.shields.io/badge/Windows-AMD%20RX%209000%20%C2%B7%20R9700-ED1C24?style=for-the-badge&logo=amd&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-windows-hip.exe)
[![Windows, AMD RX 7000 ou Ryzen AI Max](https://img.shields.io/badge/Windows-AMD%20RX%207000%20%C2%B7%20Ryzen%20AI%20Max-B3131B?style=for-the-badge&logo=amd&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-windows-hip-rdna3.exe)

**Linux**

[![Linux, NVIDIA, driver 580 ou mais novo](https://img.shields.io/badge/Linux-NVIDIA%20%C2%B7%20driver%20580%2B-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-nvidia-cuda13.AppImage)
[![Linux, NVIDIA, driver mais antigo](https://img.shields.io/badge/Linux-NVIDIA%20%C2%B7%20older%20driver-4C7A00?style=for-the-badge&logo=nvidia&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-nvidia-cuda12.AppImage)
[![Linux, AMD RX 9000 ou R9700](https://img.shields.io/badge/Linux-AMD%20RX%209000%20%C2%B7%20R9700-ED1C24?style=for-the-badge&logo=amd&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-amd-rx9070.AppImage)
[![Linux, AMD RX 7000](https://img.shields.io/badge/Linux-AMD%20RX%207000-B3131B?style=for-the-badge&logo=amd&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-amd-rx7900.AppImage)
[![Linux, AMD Ryzen AI Max](https://img.shields.io/badge/Linux-Ryzen%20AI%20Max-8A0E15?style=for-the-badge&logo=amd&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-amd-strixhalo.AppImage)

**Mac** (Apple Silicon, macOS 15 ou mais novo)

[![Mac, Apple Silicon](https://img.shields.io/badge/Mac-Apple%20Silicon-333333?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-mac.dmg)

Driver NVIDIA anterior ao 580 (o `nvidia-smi` mostra)? Use o arquivo para driver mais antigo. Uma placa AMD RX 6000 usa o arquivo
da RX 7000, que busca sozinho o mecanismo certo. Todos os arquivos estão na
[página de releases](https://github.com/Niko1221/Strata/releases/latest).

| Primeira inicialização | O que fazer |
| --- | --- |
| **Windows** | Execute o `.exe`. Não precisa de direitos de administrador. Se o Windows disser "O Windows protegeu o computador", escolha **Mais informações** e depois **Executar assim mesmo**. |
| **Linux** | Torne o arquivo executável (`chmod +x`) e inicie. É preciso ter o FUSE 2: `sudo apt install libfuse2t64` (Ubuntu 24.04 e mais novos) ou `libfuse2` (22.04). |
| **Mac** | Abra o `.dmg` e arraste o Strata para Aplicativos. |

O download do modelo é grande (40 a 70 GB). Se parar, continua de onde parou.

### 2. O Terminal (uma linha)

O comando `strata`, para scripts, servidores e agentes de IA. É também o jeito de rodar o DeepSeek.

Linux e Mac:

```sh
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

Windows (PowerShell):

```powershell
irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

Depois inicie. Ele faz uma pergunta por vez (modelo, contexto, imagens, quem pode se conectar), baixa o que precisa e começa
a servir:

```sh
strata launch
```

Não precisa de direitos de administrador. A instalação confere o SHA-256 de tudo que baixa. No Windows, abra uma nova janela do
terminal depois da instalação. Todas as telas e flags: [docs/TERMINAL_LAUNCH.md](docs/TERMINAL_LAUNCH.md).

### 3. Deixe a sua IA configurar

Cole isto no Claude Code, Cursor, Codex, GitHub Copilot ou outro assistente de programação:

```text
Set up Strata on this PC for me: https://github.com/Niko1221/Strata - follow docs/AI_SETUP.md in that repository.
```

Ele confere a sua placa de vídeo, a RAM e o disco, escolhe o modelo que cabe, instala e inicia, e diz como conectar os seus
aplicativos. Ferramentas de IA também podem instalar, iniciar e parar o Strata pelo [servidor MCP](docs/MCP_SERVER.md) dele.

Prefere compilar por conta própria? Veja [Compilar a partir do código-fonte](#compilar-a-partir-do-código-fonte).

## Modelos

O app e o `strata launch` recomendam um tamanho para a sua memória. Os tamanhos menores são mais rápidos. Os maiores são um
pouco mais inteligentes.

### Pouca memória? Experimente o Gyro

O [Gyro](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF), da agentionai, é a menor forma de rodar o
Qwen3.8-Flash-Next. Ele roda em NVIDIA, AMD, Intel Arc e Apple Silicon.

| Tamanho | Memória necessária (RAM + VRAM) | Download | Serve para |
| --- | ---: | ---: | --- |
| **Gyro-S** | 27,6 GB | 58,5 GB | PCs onde nenhum outro tamanho cabe. 9 tokens/s num Mac mini de 32 GB. |
| **Gyro-M** | 35 GB | 92,0 GB | Qualidade próxima à do IQ3_XXS, com 20% menos memória da GPU. |

Inicie com `strata launch --model gyro-s`. Detalhes: [docs/GYRO.md](docs/GYRO.md).

### Os tamanhos do Qwen3.8-Flash-Next

| Sua memória (RAM + VRAM) | Escolha | Por quê |
| --- | --- | --- |
| **32 GB** | **Gyro-S**, ou o **[Coder](docs/MODELS.md#coder)** para código | O Coder mantém os experts que o código usa |
| **40 GB** | **[Swift 1.5](docs/MODELS.md#swift-15)** IQ2_XS (39,2 GB) ou **Gyro-M** | O Swift pensa por menos tempo antes de responder |
| **48 GB** | **IQ3_XXS** (47,0 GB) | Mais lento, melhor |
| **64 GB** | **IQ3_S** (54,8 GB) | Empata com o modelo completo nos testes publicados |
| **96 GB ou mais** | **UD-IQ4_XS** a **Q8_0** da Unsloth | O mais perto do modelo completo |

Com menos RAM do que a linha indica, ainda pode funcionar, só que mais devagar: o Strata muda sozinho para o
[modo de pouca RAM](docs/MODELS.md#a-big-graphics-card-and-little-ram). Há também:
[NVFP4](docs/NVFP4.md) para placas NVIDIA, [Qwen3.6-35B-A3B](docs/QWEN36.md) (menor, vários pedidos ao mesmo tempo) e
[o Uncensored IQ3_XXS da OrcaRouter](docs/MODELS.md#orcarouter-uncensored-iq3_xxs). Todos os tamanhos e o que cabe em cada configuração:
[docs/MODELS.md](docs/MODELS.md).

### DeepSeek V4.1-Flash (Terminal, a partir da 0.2.4)

Um modelo bem maior, que roda numa única placa de vídeo de 12 GB com 46 a 64 GB de RAM. O Strata mantém na memória as partes mais
usadas e lê o resto do seu SSD. A [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF)
fez os conjuntos. Ele roda em placas NVIDIA RTX das séries 30 a 50 e na Radeon AI PRO R9700. Escolha no `strata launch`. O
app de desktop ainda não o oferece.

| Conjunto | Download | Qualidade | Escreve código a |
| --- | ---: | --- | --- |
| **IQ1_M** (1,75 bits) | 239 GB | O menor, mais fraco em prosa simples | ainda não medido num PC |
| **IQ2_XXS** (2.00 bits) | 257 GB | Bom equilíbrio | 10 tokens/s (RTX 3060), 5 (RTX 5070), 48 (4x R9700) |
| **IQ2_XS** (2.25 bits) | 274 GB | O mais perto do original | 15 tokens/s (uma R9700), 80 (4x R9700) |
| **IQ4_NL** (tabela Engram) | incluída em cada download acima | A tabela de consulta que os três conjuntos compartilham | |

Inicie com `strata launch --model ist-das-2.00bpw` (ou `ist-das-1.75bpw`, `ist-das-2.25bpw`). Números de qualidade,
velocidades por PC e a instalação manual: [docs/DEEPSEEK41.md](docs/DEEPSEEK41.md).

## Qual é a velocidade?

Com base numa placa de vídeo de 12 GB (RTX 5070, 64 GB de RAM). Veja resultados de outras placas:
[velocidade de cada tamanho](docs/MODELS.md#how-fast-is-each-size) e [resultados da comunidade](docs/COMMUNITY_BENCHMARKS.md).

| Modelo | As respostas aparecem a | Lê o seu prompt a |
| --- | --- | --- |
| **Q2_0** | `████████████` 94 tokens/s | `████████████` 2.650 tokens/s |
| **IQ2_XS** | `██████████░░` 79 tokens/s | `█████████░░░` 2.090 tokens/s |
| **IQ3_XXS** | `████████░░░░` 62 tokens/s | `████████░░░░` 1.750 tokens/s |
| **IQ3_S** | `███████░░░░░` 53 tokens/s | `███████░░░░░` 1.620 tokens/s |
| **Coder** | `███████░░░░░` 55 tokens/s | `██████████░░` 2.180 tokens/s |

Numa placa AMD de 16 GB (RX 9070 XT, 47 GB de RAM):

| Modelo | As respostas aparecem a | Lê o seu prompt a |
| --- | --- | --- |
| **Q2_0** | `████████░░░░` 60 tokens/s | `█████░░░░░░░` 1.160 tokens/s |
| **IQ2_XS** | `███████░░░░░` 52 tokens/s | `█████░░░░░░░` 1.110 tokens/s |
| **Coder** | `██████░░░░░░` 44 tokens/s | `██████░░░░░░` 1.420 tokens/s |

Um token equivale a cerca de ¾ de palavra, então 60 tokens por segundo é mais rápido do que você lê. "Lê o seu prompt" é a velocidade com que o Strata
absorve um documento longo, um código ou um histórico de conversa.

## Como usar

### No app

Um chat, um **Monitor** do modelo e da sua GPU, CPU e RAM, uma página de **Modelos** e as **Configurações**. O chat pode
trabalhar numa pasta do seu PC. No modo **Manual**, ele pergunta antes de alterar um arquivo ou executar um comando. Ele pode
pesquisar na web (DuckDuckGo ou Bing, sem conta). Os termos da pesquisa são a única coisa que sai do seu PC.

<p align="center"><img src="docs/media/app-chat.png" width="900" alt="O app Strata: um chat em que o agente montou um pequeno jardim de pagodes em voxels, com a página aberta na pré-visualização à direita"><br>
<sub>O agente escreveu um jardim de pagodes em voxels em um único arquivo HTML, e a página abre na pré-visualização ao lado do chat</sub></p>

### No Terminal

<p align="center"><img src="docs/media/terminal-monitor.png" width="900" alt="A visão geral do monitor do Terminal do Strata enquanto o modelo escreve uma resposta"><br>
<sub><code>strata monitor</code> enquanto o modelo responde: velocidade, GPU, memória e as solicitações recentes</sub></p>

```sh
strata launch                              # guided: model, context, pictures, access, then serve
strata launch --model q2_0 --yes --no-tui  # no questions: serve and print the address
strata launch --status                     # what is being served now
strata launch --stop                       # stop it
strata monitor                             # watch speed, GPU, memory and requests
strata update                              # get the newest engine
strata doctor                              # check the installed files
```

Para acessar o Strata do celular ou de outro PC, use `strata launch --access lan`. Ele cria uma chave de API para você.
`strata --help` lista tudo, e o [docs/TERMINAL.md](docs/TERMINAL.md) explica.

### Com agentes de IA e outros aplicativos

Um agente instala o Strata com o comando de uma linha acima e executa `strata launch --model NAME --yes --no-tui`. Os passos,
as flags e os códigos de saída estão em [docs/AI_SETUP.md](docs/AI_SETUP.md). As regras do repositório para agentes: [AGENTS.md](AGENTS.md).

| Seu aplicativo fala | URL base |
| --- | --- |
| API da OpenAI (a maioria dos apps e agentes) | `http://127.0.0.1:8080/v1` (qualquer chave de API, qualquer nome de modelo) |
| API da Anthropic (Claude Code) | `http://127.0.0.1:8080`, sem `/v1`. Claude Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:8080` |
| OpenAI Responses API (Codex CLI) | `http://127.0.0.1:8080/v1/responses` ([configuração](docs/DETAILS.md#the-responses-api-and-codex-cli)) |

`strata connect` mostra as configurações exatas para OpenCode, Codex CLI, Claude Code, oh-my-pi, pi, Hermes Agent e OpenClaw,
e `strata connect opencode --write` grava para você (com um backup).

### Bom saber

- **As imagens** vêm ativadas em todo modelo que sabe lê-las. Anexe ou cole uma imagem no chat, ou envie pelo seu aplicativo. Detalhes: [docs/IMAGES.md](docs/IMAGES.md).
- **O raciocínio** pode ser off, low, medium ou high, no menu do chat ou no "reasoning effort" do seu aplicativo. Off é o mais rápido.
- **A primeira mensagem** de uma conversa é lida por inteiro, cerca de 1 minuto a cada 30.000 tokens. As mensagens seguintes começam em segundos.
- **Um pedido por vez** é o padrão. Para mais de um, use `--parallel N` ([BATCHING.md](docs/BATCHING.md)). Numa placa de 12 GB, isso deixa cada resposta mais lenta.

## Compilar a partir do código-fonte

<details>
<summary>Windows, Linux e Mac a partir do repositório</summary>

[Baixe o Strata](https://github.com/Niko1221/Strata/archive/refs/heads/main.zip) e descompacte, ou use `git clone`.
**Windows:** dê dois cliques em `START-HERE.bat`. **Linux:** execute `./setup.sh` na pasta do Strata.

**Mac (Apple Silicon, macOS 15 ou mais novo):** instale uma vez as Command Line Tools da Apple com `xcode-select --install`
(o Xcode em si não é necessário). Abra o Terminal na pasta do Strata e execute `./setup.sh`. Ele compila o mecanismo do Mac no seu
Mac (cerca de 7 minutos), faz as mesmas perguntas, baixa o modelo e inicia. Da próxima vez, `./setup.sh` inicia
na hora. Qual tamanho cabe no seu Mac: [docs/MAC.md](docs/MAC.md).

O instalador encontra a sua placa e configura o mecanismo certo. Ele pergunta qual modelo e tamanho, quanto contexto (quanto texto
o modelo mantém em mente) e se deve ler imagens. Pressione Enter para aceitar a resposta recomendada. Depois ele baixa o
modelo (cerca de 70 GB) e inicia. Se o download parar, execute de novo e ele continua. O navegador abre a página do Strata em
`http://127.0.0.1:8080`.

> Enquanto o modelo inicia, o seu PC pode ficar lento ou parar de responder por 1 a 3 minutos (mais tempo na primeira vez). O Strata carrega
> de 35 a 55 GB na sua RAM e trava parte dela para a placa de vídeo. Espere e não feche a janela.

Da próxima vez, execute `START-HERE.bat` (ou `./setup.sh`) de novo. Ele inicia na hora e não baixa nada duas vezes. Feche a janela
dele para parar o modelo. `UPDATE.bat` (`./update.sh`) atualiza o Strata sem iniciá-lo. Atualização, Docker, várias placas,
onde ficam os arquivos e todas as opções: [docs/INSTALL.md](docs/INSTALL.md).

Para medir o seu PC, execute `BENCH.bat` (`./bench.sh`). Leva de 5 a 10 minutos e gera um relatório que você pode compartilhar
([como](docs/COMMUNITY_BENCHMARKS.md#run-the-benchmark)).

**Opcional: mais velocidade numa placa quando o modelo inteiro cabe na RAM.** Duas configurações opcionais do RubenTorney (#1904, #1905).
Elas ajudam quando você usa uma placa de vídeo e a sua RAM comporta todos os experts, por exemplo Q2_0 ou IQ3_XXS num PC de 64 GB. Abra
`strata-<model>.json` na pasta de dados do Strata ([onde fica](docs/INSTALL.md#where-things-are-stored)) e adicione:

- `"--adapt-async", "1"` à lista `"args"`. O cache de experts passa a ser atualizado numa thread auxiliar, em vez de pausar a
  resposta. Medido numa RTX 5070 (Windows, Q2_0): +4 a 5% na geração, +7 a 10% em código e documentos longos. As respostas podem diferir
  um pouco do padrão e continuam iguais de uma execução para outra.
- `"STRATA_PIN_BY_NEED": "1"` no bloco `"env"` (Windows). O Windows só consegue travar cerca de metade da RAM para a placa
  de vídeo, e isso trava primeiro os experts de que a placa precisa. Medido na RTX 5070: +4,4% na leitura do prompt.

```json
"args": [ ..., "--adapt-async", "1" ],
"env": { "STRATA_PIN_BY_NEED": "1" }
```

Reinicie o Strata depois da alteração. Para desfazer, remova as duas entradas.

</details>

## Como funciona?

Modelos como este costumam rodar em servidores com centenas de gigabytes de memória de vídeo. A sua placa de vídeo tem
12 a 24 GB. O Strata faz o modelo caber dividindo o trabalho entre o PC inteiro. Imagine uma biblioteca.

- O modelo é uma equipe de 24.576 pequenos especialistas ("experts"), como 24.576 livros de consulta. Cada palavra precisa de apenas 10 deles.
- **A sua placa de vídeo é a mesa.** Ela é pequena e rápida. Os cerca de 4.000 livros mais usados ficam abertos nela.
- **A sua RAM são as estantes da sala de leitura.** Os 24.576 livros estão lá. Quando uma palavra precisa de um livro que não está na
  mesa, o seu processador lê da estante na hora, enquanto a placa continua trabalhando.
- **O seu SSD é o arquivo.** Ele guarda uma tabela de consulta de 29 GB. O modelo lê algumas linhas dela por palavra, então ela não
  precisa ficar na RAM.

<p align="center"><img src="docs/media/how-it-works.svg" width="860" alt="Os 24.576 experts do modelo: os mais usados na placa de vídeo, todos na RAM, uma tabela de consulta no SSD"></p>

Um pequeno ajudante também adivinha as próximas palavras. O modelo grande confere todas de uma vez e fica com as certas. Você recebe
a mesma resposta, de 1,6 a 1,8 vez mais rápido. Textos longos são lidos em blocos grandes, de até 8.192 tokens por vez, a mais de 1.000 tokens
por segundo.

<p align="center"><img src="docs/media/guess-and-check.svg" width="860" alt="Um pequeno ajudante adivinha as próximas palavras; o modelo grande confere todas de uma vez e fica com as certas"></p>

A explicação mais longa: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md). Cada parte e seus números:
[os detalhes](docs/DETAILS.md#how-it-works) e o [artigo](docs/paper/Strata-Paper.pdf).

## Algo deu errado?

- **Meu PC travou na primeira vez que o Strata iniciou.** Ele carrega o modelo na RAM. Espere e não feche a janela. Ainda
  travado depois de 10 minutos? Reinicie o PC, feche outros programas e tente de novo, ou escolha um tamanho menor.
- **Parou durante o download ou a instalação.** Abra o app ou o `strata launch` de novo. Ele continua de onde parou.
- **Está muito lento, ou aparece "the engine stopped unexpectedly".** O seu PC não tem RAM livre suficiente. Feche outros
  programas (os navegadores usam muita), ou escolha um tamanho menor, como Q2_0 ou IQ2_XS.
- **Diz que não há mecanismo para este PC.** No app, abra **Configurações, Mecanismo** e clique em **Atualizar mecanismo**. No
  Terminal, execute `strata update`.
- **Diz que a porta 8080 já está em uso.** O Strata já está rodando. Procure a janela dele.

Mais soluções: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). Ainda sem resolver? Abra uma
[issue](https://github.com/Niko1221/Strata/issues) e anexe os logs: no app, em **Configurações, Mecanismo, Diagnóstico,
Salvar logs**; no Terminal, a saída de `strata doctor`. Para um problema de segurança, informe em particular: [SECURITY.md](SECURITY.md).

## Créditos e licença

O modelo é o [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), da equipe Qwen. Ele foi comprimido por
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), UkisAI (Swift 1.5), agentionai (Gyro)
e Unsloth. A ISTA-DASLab também fez os conjuntos do DeepSeek V4.1-Flash. O Strata usa partes do
[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). Todos os créditos: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits).
O Strata é de código aberto sob a [Licença MIT](LICENSE). Algumas partes e todos os modelos têm licenças próprias
([quais](docs/HOW_IT_WORKS.md#license)).

## Contribuindo

Encontrou um bug, ou quer enviar uma alteração ou um benchmark? Leia primeiro o [CONTRIBUTING.md](CONTRIBUTING.md). Ele diz o que
incluir para que outras pessoas possam conferir o seu trabalho.

## Apoie o Strata

O Strata é gratuito e tem licença MIT. Eu o desenvolvo e testo em todo hardware que consigo arranjar. Se ele roda bem no seu PC, um café me ajuda
a continuar testando placas novas e corrigindo o que você relatar. Sem pressão. Usar e relatar o que encontrar ajuda tanto quanto.

<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a>

Quer patrocinar o Strata? Fale comigo pelo [GitHub](https://github.com/Niko1221).
