# Qwen3.8-Flash-Next IQ3_XXS, CPU reference output

Produced on <服务器地址> (2x910B2C, openEuler 24.03, CANN 9.0.0) with llama.cpp master built
`GGML_CANN=OFF`, `GGML_NATIVE=OFF`, Release. Single thread and a fixed seed, so the reduction
order, and therefore the text, are reproducible.

```
llama-cli -m Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
    -ngl 0 -t 1 -c 512 -n 16 --seed 1 -p "用一句话说明什么是混合专家模型"
```

Measured with that command: prompt 3.3 t/s, generation 1.4 t/s on one thread. With eight threads
the same prompt gave 18.9 t/s and 7.6 t/s.

A port is correct when it produces this text for this prompt and seed. Anything else means the
layers are not matching.

## Output

```
> 用一句话说明什么是混合专家模型

[Start thinking]

我们需要回答用户：用一句话说明什么是混合专家模型。需要中文。工具
```

The run stops mid-sentence because `-n 16` cut it there.

## Why this is the only reference we have

llama.cpp's CANN backend cannot run this model on this box yet. In order: `Repeat` fails to
launch its kernel (`BinaryGetFunction failed, kernel_name=`), and with `Repeat` on the CPU
`ScatterUpdate` faults on the aicore. See section 11.9 of `ASCEND_910B_PORT_STATUS.md`.

So the CPU run above is the only ground truth. It is enough to check a port layer by layer, which
is what matters: a port that reproduces this text has the maths right.
