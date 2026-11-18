# Hakua / Strata

[English](README.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [Français](README.fr.md) · [Polski](README.pl.md) · [Português](README.pt-BR.md) · [简体中文](README.zh-CN.md)

IA local para Windows com chat, trabalho de agente, pesquisa com fontes e escolha de modelos compatíveis. Hakua é o fork do zapabob baseado em Niko1221/Strata; é mantido de forma independente e não é uma distribuição endossada pelo projeto original.

## Versão e downloads

<!-- release-062-stable -->
Versão atual: [0.6.3](https://github.com/zapabob/Strata/releases/tag/v0.6.3) (App `0.6.3`, Native Q38 `0.6.3` SM120, Control `0.6.3`). `releaseQualified=false` permanece explícito; inferência em janela completa e qualidade geral de modelos seguem não qualificadas.

[0.6.3](https://github.com/zapabob/Strata/releases/tag/v0.6.3) · [Strata-0.6.3-windows-x64-setup.exe](https://github.com/zapabob/Strata/releases/download/v0.6.3/Strata-0.6.3-windows-x64-setup.exe) · [tar.7z](https://github.com/zapabob/Strata/releases/download/v0.6.3/Strata-0.6.3-windows-x64.tar.7z) · [SHA256SUMS](https://github.com/zapabob/Strata/releases/download/v0.6.3/SHA256SUMS.txt)

![Hakua / Strata real-model Japanese chat on Qwen3.8-27B](docs/media/native-qwen38-japanese-chat.png)
*Hakua / Strata Desktop executando `qwen3.8-27b-abliterated-3.69bpw-12GB-MTP.gguf` via `strata-q38.exe` (CUDA 13 / RTX 5060 Ti 16 GB).*

## Instalação e atualização

Use o instalador NSIS para Windows x64 deste fork. Ele instala por usuário e inclui Node, Python e Control privados; ferramentas de desenvolvimento do sistema não são necessárias. O motor CUDA nativo é um componente separado. O pacote `tar.7z` reúne instalador, pacote do motor CUDA, pacote Control, instruções e checksums, excluindo pesos de modelos e dados pessoais.

Conclua solicitações ativas, descarregue modelos, faça backup dos dados e feche o app incluindo o ícone da bandeja. Execute o instalador na pasta existente para atualizar por sobrescrita preservando modelos, conversas, memória, ferramentas, chaves e aprovações. Importe o componente CUDA compactado com *Já tenho o arquivo*, escolha um modelo compatível ou um GGUF local e carregue-o. Veja [instalação e reversão](docs/HAKUA_INSTALL.md).

## Novidades da 0.6.2 e recursos do fork

A versão 0.6.2 aprimora o [executor local GGUF denso da família Qwen3.8-27B (`strata-q38.exe`)](docs/QWEN38_DENSE_NATIVE.md) no Windows CUDA 13 / Blackwell (SM120):
- **Polimorfismo de quantização por tensor (`StrataQuantTensor`)**: suporta 11 tipos GGML/Strata (`F32`, `F16`, `BF16`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ3_S`, `IQ2_S`, `IQ4_XS` e PrismML `PQ2_0=142`) por tensor.
- **Prefill em lote Tensor Core + GDN multi-token + Atenção Spin8 fundida + Decode com CUDA Graph (CUDA 13.5 SM120)**:
  - Prefill de 140 tokens (`--kv spin8_k8v4 --ctx-size 4096`): caiu de `36.760,7 ms` (`3,81 tok/s`, `--conservative-sync`) para **`1.558,3 ms` (`89,8 tok/s` aquecido, aceleração de 23,6x)**.
  - Decode contínuo (`GEN 32`, 31 passos, GEMV empacotado + CUDA Graph com barreira a cada 8 camadas): caiu de `206,4 ms/passo` para **`85,13 ms/passo` (`11,75 passos/s`, aceleração de 2,42x)**.
  - **Medições de contexto longo (`--kv spin8_k8v4`, RTX 5060 Ti 16 GB)**: **65K (`65.536`)** aprovado (prefill de 1.024 tokens `10.813,5–11.220,7 ms`, `91,3–94,7 tok/s`; decode `670,4 ms` / 7 passos); **96K (`98.304`)** aprovado (`required_with_reserve=15,09 GB`, prefill de 140 tokens `1.968,7 ms`); **131K (`131.072`)** bloqueado pela reserva de 768 MiB da GPU (`required=15,97 GB > available=15,91 GB`).
- **Verificação real de rascunho MTP `K=3` (`blk.64.nextn.*`)**: suporta `--spec-draft-n-max 0..3` com `KvCache::rewind` e reversão de estado GDN.
- **Cache KV e perfil japonês**: modos `F16`, `Spin8 K8V4` e `F16-K / Turbo2/3/4-V` (ordem `auto`: `f16` → `f16_turbo4` → `spin8_k8v4`), além de perfil opcional de saída em japonês.

`/btw` inicia um subagente real sem ferramentas para perguntas paralelas com controle de parada próprio. `/research` e `/deepresearch` coletam fontes usando o RustXNG Full integrado (consultas vão para provedores externos habilitados). Aprendizado e aplicação automática de habilidades de fluxo são opções separadas e desativadas por padrão em novos perfis. Veja [habilidades](docs/WORKFLOW_SKILLS.md) e [busca](docs/BROWSER_SEARCH.md).

## Integração oficial e compatibilidade

A base oficial fixada é [c775728069](https://github.com/Niko1221/Strata/commit/c77572806996349b49995e4f98d41b5294d36a21), associada ao 0.2.15 / motor 0.1.60. As versões do fork são independentes. A compatibilidade com DeepSeek é mantida como extensão e impede o carregamento sem o runtime dedicado. Este pacote Windows não implica entrada nativa de imagem, motor DeepSeek nativo distribuído ou inferência qualificada em Linux/Mac.

## Proteções e limites medidos

Mantenha os pisos de 3 GiB de RAM disponível, 4 GiB de margem de commit, pelo menos 768 MiB de reserva de GPU e limites de disco. 32 GB de RAM e 16 GB de VRAM não garantem que todo modelo ou contexto caiba. `releaseQualified=false` permanece válido e o instalador não possui assinatura Authenticode do Windows. Medições históricas permanecem em [evidências de lançamento](docs/HAKUA_RELEASE.md).

## Documentação e desenvolvimento

[Instalação](docs/HAKUA_INSTALL.md) · [Evidências](docs/HAKUA_RELEASE.md) · [API e hardware](docs/AI_REFERENCE.md) · [Terminal](docs/TERMINAL.md) · [Compilação](docs/SOURCE_INSTALL.md) · [Empacotamento](packaging/README.md) · [Contribuir](CONTRIBUTING.md) · [Segurança](SECURITY.md) · [Licença](LICENSE)

```sh
python -m pytest tests/terminal
```

```sh
cd apps/host
pnpm test
pnpm run build
```

### Niko1221/Strata — referência oficial de Terminal

```sh
curl -fsSL --connect-timeout 15 --max-time 120 https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

```powershell
irm -TimeoutSec 120 https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

[Niko1221/Strata](https://github.com/Niko1221/Strata) · [Hermes Agent](https://github.com/NousResearch/hermes-agent) · [Fontes e avisos de terceiros](third_party/)

## O que mudou na 0.6.3

- O prefill nativo do Q38 usa GEMM de Tensor Core do cuBLAS com ativações FP16 escaladas por token e um bloco de pesos persistente em L2 de 20 MiB. O decode usa GEMV dp4a Q8_1 alinhado conforme o formato e um CUDA Graph com LayerFence(8).
- A admissão de contexto longo escolhe o tamanho do bloco de prefill a partir de uma escala de VRAM medida. A reserva de GPU de 768 MiB e os mínimos de 3 GiB de RAM e 4 GiB de commit permanecem inalterados.
- Os fixtures numéricos normalizam os pesos compactados sintéticos para amplitudes realistas sem afrouxar as tolerâncias rigorosas de `_exact_q8`. O GEMV desalinhado recorre a um kernel escalar de leitura por byte.
- Verificação: 17/17 testes nativos passaram; compute-sanitizer relata 0 erros e 0 riscos em memcheck/synccheck/racecheck. A qualidade geral do modelo, a inferência em janela completa e a concorrência de GPU continuam não qualificadas (`releaseQualified=false`).
- Base de pesquisa: notas de versão do NVIDIA CUDA 13.5 (correção do algoritmo 66 do cuBLASLt com Tensor Memory) e o relatório de arquitetura Qwen3.8-Next (híbrido GDN/atenção global 3:1, QSA com compressão r=4 e orçamento K=2048, residual com porta, embedding de n-gramas pré-carregado no host).
