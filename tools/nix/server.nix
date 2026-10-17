{
  lib,
  stdenvNoCC,
  python3,
  makeWrapper,
  src,
}:

let
  python = python3.withPackages (
    ps: with ps; [
      numpy
      jinja2
      regex
      pyyaml
      tqdm
      requests
      pillow
      psutil
    ]
  );
in
stdenvNoCC.mkDerivation {
  pname = "strata-server";
  version = "0.2.20";
  inherit src;

  strictDeps = true;
  nativeBuildInputs = [ makeWrapper ];
  nativeCheckInputs = [ python ];
  nativeInstallCheckInputs = [ python ];
  dontConfigure = true;
  dontBuild = true;

  doCheck = true;
  checkPhase = ''
    runHook preCheck
    export PYTHONDONTWRITEBYTECODE=1
    (cd terminal && ${python}/bin/python -m unittest \
      serve.test_frontend serve.test_runconfig serve.test_security \
      serve.test_server serve.test_structured serve.test_logit_bias)
    runHook postCheck
  '';

  installPhase = ''
    runHook preInstall
    app="$out/share/strata"
    mkdir -p "$app/third_party" "$out/bin"
    # The Terminal (0.2.16 layout: terminal/serve, terminal/tools) keeps its upstream relative layout.
    # setup.py, strata_cli and strata_control are part of the runtime: serve/ and the tools import them.
    cp -r terminal/serve terminal/tools terminal/strata_cli terminal/strata_control terminal/setup.py "$app/"
    find "$app" -type d -name __pycache__ -prune -exec rm -rf {} +
    # The vendored gguf-py the tools look for at <app>/third_party/strata-gguf-py (tools/_paths.py).
    # Copy only that folder so the runtime closure does not retain the full C++ tree.
    cp -r engine/third_party/strata-gguf-py "$app/third_party/strata-gguf-py"
    # No chdir, venv, setup.py, downloads, or dynamic-loader environment variables.
    makeWrapper ${python}/bin/python "$out/bin/strata-server" \
      --add-flags "$app/serve/server.py"
    for tool in iq_pack mtp_pack mtp_rt mtp_fetch strata_tokenizer; do
      name="''${tool#strata_}"
      name="strata-''${name//_/-}"
      makeWrapper ${python}/bin/python "$out/bin/$name" \
        --add-flags "$app/tools/$tool.py"
    done
    install -Dm644 LICENSE "$out/share/licenses/strata/LICENSE"
    runHook postInstall
  '';

  doInstallCheck = true;
  installCheckPhase = ''
    runHook preInstallCheck
    test -z "''${LD_LIBRARY_PATH-}"
    ${python}/bin/python ${./test-package.py} "$out" --server-only
    runHook postInstallCheck
  '';

  meta = {
    description = "Strata API server, web app and model preparation tools";
    homepage = "https://github.com/Niko1221/Strata";
    license = lib.licenses.mit;
    platforms = lib.platforms.linux;
    mainProgram = "strata-server";
  };
}
