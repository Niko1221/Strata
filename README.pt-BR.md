## O que muda em 0.7.5

Nota: os valores DS4 2,84x / FLA 4,86x / Sakura 22,44 t/s vêm dos logs de desenvolvimento do Antigravity e não foram reproduzidos de forma independente nesta sessão de build. Esta sessão mediu 9,09 t/s de prefill / 11,78 t/s de decode com o GGUF canônico a 32K.

Esta versão mantém todos os recursos do fork e adiciona as otimizações nativas de 22/11: produto **0.7.5**, família de mecanismos **0.6.7**, upstream v0.2.19. RTX 5060 Ti (sm_120a)/CUDA 13.5: DS4-9B prefill 2,84x, E2E +22,7 %, FLA GDN + Spin(8) prefill 4,86x, Sakura Inkling-2 65K-98K (22,44 t/s decode). Tabelas SM x VRAM de Ryu (kuraneko1, MIT) no Inkling-2 CUDA oficial (PR #6571). Verificação de assinatura removida; SHA-256 permanece. `releaseQualified=false`.

## Versão e downloads

<!-- release-062-stable -->
Versao atual do aplicativo: [0.7.5](https://github.com/zapabob/Strata/releases/tag/v0.7.5) (App `0.7.3`; Terminal, Control `0.6.3`; familia de mecanismo nativo `0.6.5`) — este e o GitHub Latest. O aplicativo Windows inclui a familia de mecanismos 0.6.5 (Q38 27B+9B recompilado, Hy4-Flash 0.1.64 oficial) como componentes opcionais separados. A verificacao de assinatura foi removida (sem chave); os hashes SHA-256 permanecem como controle de integridade. `releaseQualified=false` permanece; inferencia de janela completa e qualidade do modelo nao qualificadas. Um terceiro mecanismo opcional, `strata-inkling2` (Inkling-2 Mini, arch GGUF `inkling2`), foi portado de kuraneko1/Strata (MIT) por kuraneko1 (_ryu15_) com credito.

[0.7.5](https://github.com/zapabob/Strata/releases/tag/v0.7.5) · [EXE de instalação](https://github.com/zapabob/Strata/releases/download/v0.7.5/Strata-0.7.5-windows-x64-setup.exe) · [tar.7z](https://github.com/zapabob/Strata/releases/download/v0.7.5/Strata-0.7.5-windows-x64.tar.7z) · [SHA256SUMS](https://github.com/zapabob/Strata/releases/download/v0.7.5/SHA256SUMS.txt) · [BUNDLE-SHA256SUMS](https://github.com/zapabob/Strata/releases/download/v0.7.5/BUNDLE-SHA256SUMS.txt)

![Hakua / Strata real-model Japanese chat on Qwen3.8-27B](docs/media/native-qwen38-japanese-chat.png)
*Hakua / Strata Desktop executando `qwen3.8-27b-abliterated-3.69bpw-12GB-MTP.gguf` via `strata-q38.exe` (CUDA 13 / RTX 5060 Ti 16 GB).*

## GGUF recomendado para o protótipo Qwen3.8

Para testar conversas de texto locais numa RTX 5060 Ti com 16 GB de VRAM, recomendamos o [GGUF com quantização mista de 3,69 bpw de soyaakinohara](https://huggingface.co/soyaakinohara/qwen3.8-27b-abliterated-3.69bpw-12GB-MTP.gguf) usado nos nossos testes. Agradecemos a **soyaakinohara** pela quantização e publicação e à **AEON-7** pelos pesos de origem. Baixe o modelo separadamente e comece com um contexto de 4.096 tokens. O nome do arquivo não garante funcionamento com 12 GB de VRAM.

A [prévia nativa assinada para CUDA 13.5](https://github.com/zapabob/Strata/releases/tag/v0.7.2) passou em dois turnos em japonês, na interrupção da resposta e na remoção do modelo da memória no aplicativo 0.6.3. Em dois testes da primeira solicitação, o tempo de prefill F16 curto caiu 35–43% em relação ao caminho nativo anterior; Prism (llama.cpp) ainda decodificou mais rápido. [Configuração, chave pública e limites](docs/Q38_NATIVE_PREFILL_PREVIEW.md).

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

A base oficial fixada é [6e38f588](https://github.com/Niko1221/Strata/commit/6e38f588a646c0ee6948a74afbe670eb75896106), associada ao 0.2.15 / motor 0.1.60. As versões do fork são independentes. A compatibilidade com DeepSeek é mantida como extensão e impede o carregamento sem o runtime dedicado. Este pacote Windows não implica entrada nativa de imagem, motor DeepSeek nativo distribuído ou inferência qualificada em Linux/Mac.

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
