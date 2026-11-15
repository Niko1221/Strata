# Hakua / Strata

[English](README.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [Français](README.fr.md) · [Polski](README.pl.md) · [Português](README.pt-BR.md) · [简体中文](README.zh-CN.md)

IA local no Windows para conversar, trabalhar com agentes e pesquisar com fontes. Hakua é o fork de Niko1221/Strata mantido de forma independente por zapabob; não é uma distribuição endossada pelo projeto original.

## Versão e downloads

<!-- release-060-candidate -->
Este código é o candidato de integração 0.6.0. Aplicativo, Terminal, Control e fontes do motor nativo estão em 0.6.0. As verificações de compilação, instalação e publicação continuam; o número de versão não comprova a disponibilidade dos arquivos. A [versão 0.5.9](https://github.com/zapabob/Strata/releases/tag/v0.5.9) verificada foi preservada. Consulte o registro de entrega para o estado real de 0.6.0.

[0.6.0](https://github.com/zapabob/Strata/releases/tag/v0.6.0) · [Strata-0.6.0-windows-x64-setup.exe](https://github.com/zapabob/Strata/releases/download/v0.6.0/Strata-0.6.0-windows-x64-setup.exe) · [tar.7z](https://github.com/zapabob/Strata/releases/download/v0.6.0/Strata-0.6.0-windows-x64.tar.7z) · [SHA256SUMS](https://github.com/zapabob/Strata/releases/download/v0.6.0/SHA256SUMS.txt)

## Instalar e atualizar

Use o instalador NSIS Windows x64 deste fork. A instalação por usuário inclui Node, Python e Control próprios, sem exigir ferramentas de desenvolvimento no sistema. O motor CUDA assinado é um componente separado. O tar.7z deve conter o mesmo instalador, componentes, instruções e somas de verificação, sem pesos de modelos ou dados pessoais. Extraia primeiro 7z e depois tar; confira BUNDLE-SHA256SUMS.txt.

Termine as solicitações, descarregue os modelos da memória, faça backup dos dados e feche também o ícone da bandeja. Instale na pasta existente. Preserve modelos, conversas, memória, ferramentas, chaves e aprovações. Importe o motor compactado pela opção de arquivo existente, escolha um modelo compatível ou GGUF local e carregue. O nome do modelo não comprova compatibilidade. [Instalação e reversão](docs/HAKUA_INSTALL.md).

## Recursos preservados

`/btw` inicia um subagente real sem ferramentas para perguntas paralelas, com parada própria. `/research` e `/deepresearch` reúnem fontes. A busca pode usar RustXNG Full incluído. As consultas vão aos provedores externos habilitados; a busca não é totalmente offline. A quantidade de fontes não garante precisão.

Aprendizado de habilidades e aplicação automática são opções separadas, ambas desativadas em novos perfis. Preferências, habilidades privadas editáveis, revisões e desfazer são preservados. Também permanecem operações de arquivos, histórico e memória com escopo, caminhos PrismML/TurboQuant compatíveis, memória limitada de especialistas, integridade de packs e diagnósticos. [Habilidades](docs/WORKFLOW_SKILLS.md) · [Busca](docs/BROWSER_SEARCH.md).

## Integração oficial e compatibilidade

A referência oficial é [baf16363b4](https://github.com/Niko1221/Strata/commit/baf16363b4403ef0ed4f538f5d5ed11c43fb9535), geração 0.2.14 / motor 0.1.60: rede, escolha de rascunhador, kernels de prompts, memória e correções de plataforma. As versões do fork são independentes. Caminhos DeepSeek removidos são mantidos como extensão; sem o executável dedicado, a carga deve ser bloqueada. Este pacote Windows não valida entrada nativa de imagens, distribuição de motor nativo DeepSeek ou inferência Linux/Mac.

## Proteção e limites medidos

Permanecem os mínimos de 3 GiB de RAM disponível, 4 GiB de margem de memória comprometida, pelo menos 768 MiB de reserva GPU e espaço livre em disco. A proteção RAM do Windows observa carga e geração. 32 GB RAM e 16 GB VRAM não garantem qualquer modelo ou contexto. Janela completa, qualidade geral e concorrência GPU continuam sem qualificação: releaseQualified=false. O instalador não tem assinatura Windows Authenticode; assinaturas de componentes e do feed são mecanismos separados. As [medições históricas](docs/HAKUA_RELEASE.md) mantêm suas versões originais.

## Documentação e desenvolvimento

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

[Niko1221/Strata](https://github.com/Niko1221/Strata) · [Hermes Agent](https://github.com/NousResearch/hermes-agent) · [Fontes e avisos de terceiros](third_party/)
