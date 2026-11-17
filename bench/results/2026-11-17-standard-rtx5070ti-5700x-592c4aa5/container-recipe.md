# Local container packaging and launch

This is a **custom-track** deployment, not a published Strata 0.2.15 Docker image. It uses the unmodified official Control 0.1.61 archive and the CUDA13 engine 0.1.60 archive selected by the 0.2.15 update feed. Both archive SHA-256 values were verified before extraction. The engine's nine manifest file hashes were also verified after installation. No engine compilation or runtime-source patch was performed.

[Dockerfile.used](Dockerfile.used) is the actual image-layer recipe. Its base is the locally pulled `ghcr.io/niko1221/strata:0.2.13-cuda13` image, image ID `sha256:255171696992feae970f2ee178e3edb9561c7422ca37e3d09d6c9abf24b8aa3e`. The new image ID is `sha256:47139b33cc5cef628f88cde93fafb95b481ce74ed2c23be9b3e9d483cb217a06`. The 0.2.13 entrypoint/wrapper, healthcheck and system libraries were inherited; Terminal and engine directories were removed and replaced by the extracted official archives. Local file modification times were refreshed before COPY to avoid legacy Docker's same-size/epoch-timestamp layer-diff issue; file contents were unchanged.

The build context had `terminal/` containing the extracted Control archive and `engine/engine/` containing the extracted engine archive, whose `engine-component.json` is at its root. [packages.json](provenance/packages.json) records the exact download URLs, sizes and SHA-256 values. Resolve the Control relative URL against `https://github.com/Niko1221/Strata/releases/download/v0.2.15/`. Hash-check the archives, extract them into those folders, then run:

```sh
docker pull ghcr.io/niko1221/strata:0.2.13-cuda13
# Verify its local image ID matches the base recorded above.
docker build --pull=false -f Dockerfile.used -t strata-local:0.2.15 .
```

Sanitized equivalent of the serving launch (replace the example mount locations and provide your own key file):

```sh
docker run -d --name strata-benchmark --gpus all --cpus 10 \
  --shm-size 2g --stop-timeout 120 --cap-drop ALL \
  --security-opt no-new-privileges:true -p 127.0.0.1:8080:8080 \
  --mount type=bind,src=/example/data,dst=/data \
  --mount type=bind,src=/example/models,dst=/models,readonly \
  --mount type=bind,src=/example/api-key,dst=/run/secrets/strata_api_key,readonly \
  -e FAMILY=qwen -e MODEL=IQ3_XXS -e MODEL_PATH=/models/IQ3_XXS \
  -e CONTEXT=128k -e PARALLEL=1 -e KV=int8 -e VISION=cpu -e TUNE=skip \
  -e API_KEY_FILE=/run/secrets/strata_api_key strata-local:0.2.15
```

Mount locations, container name and published address above are publication-safe examples; the actual benchmark used loopback inside the container. The production published listener was on a private network with an API key and firewall. No credentials are included. Existing prepared packs and helper files were reused, with model/helper hashes in results.json. Effective resolved arguments are in config.json; they take precedence over requested launcher values. The outer Proxmox LXC had 54 GiB memory and 10 logical CPUs in both runs; Docker had no additional memory limit. No new hardware-specific retuning was run after the CPU swap.
