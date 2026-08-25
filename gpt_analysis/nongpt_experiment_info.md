需要和其他baseline比较的实验：

5.2 overall performance （Table3）`experiments/exp1_overall/run.sh`

5.5 DACI与其他baseline，验证请求长度变长时，DACI相对静态方案的优势是否扩大。（Figure 5）`experiments/exp5_scalability/G_sweep.sh`


```
outputs/
  exp1_overall_small/       # Gemma3-4B
    DACI/
    SDA/
    RT/
    FM/
  exp1_overall_medium/      # LLaMA-3.2-8B
    DACI/
    SDA/
    RT/
    FM/
  exp1_overall_large/       # Qwen3-14B
    DACI/
    SDA/
    RT/
    FM/
```


```
outputs/
  exp5_scalability/
    G_sweep/
      G_5000_DACI/
      G_5000_SDA/
      G_10000_DACI/
      G_10000_SDA/
      G_15000_DACI/
      G_15000_SDA/
      G_20000_DACI/
      G_20000_SDA/
      G_40000_DACI/
      G_40000_SDA/
```


备注：

1. `experiments/exp1_overall/run.sh` 脚本只跑4个scheme、`N_TRACES=5`，而且还写了过时模型名 `llama-3.2-8b`，和 paper 里“默认 30 traces、Llama-3-8B”不一致，所以不一定可以直接复现文中数据。
2.
