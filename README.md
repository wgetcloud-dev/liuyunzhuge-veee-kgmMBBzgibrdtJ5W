
## ChatGPT生成的文章摘要


这篇博客记录了作者在家中使用Pascal显卡运行大型模型时遇到的挑战和解决方案。随着本地大型模型性能的提升，作者选择使用vllm库进行推理。然而，作者遇到了多个技术难题，需要自行编译vllm和PyTorch，以支持Pascal架构的显卡。编译过程中，作者深入研究了显卡不支持的问题，特别是在量化矩阵乘法计算中发现性能瓶颈。最终，解决了性能问题，让性能提升了43倍。这次技术探索不仅解决了具体问题，还为作者提供了深入学习和扩展其他相关技术的机会，同时也展示了LLM在整个过程中提供的帮助。文章结尾，作者总结了经验并提出了进一步研究的方向。


## 背景


家里有张Pascal架构的显卡【划重点，后面要考】，最近发现本地大模型的性能在蹭蹭往上涨，于是开始研究下是否能在本地跑大模型。


之前我就了解vllm库，vllm的推理速度还是很快的，并且我之前还给vllm提交过一个PR，对vllm比较熟悉，所以我选择了使用vllm来进行推理。


选择结束之后就开始了漫长的抗争之路，期间着实遇到了很多问题，也学到了很多知识，故写此文以记录。


## 第一关：下载安装


当时无知的我以为安装是一件很简单的事情，以前使用vllm，直接`pip install vllm`，不仅会帮忙安装好vllm，pytorch，还会帮忙下载对应的cuda库，自己啥都不用操心。


这次的安装也如以前一样顺利，


安装完后就是选择模型了，选择模型的话，对于消费级显卡来说，显存占用是一个主要的考量因素，你得先跑起来。获取模型的显存占用的方式有两种：


1. 计算模型需要占用的显存大小，比如一个7B的模型，它的参数量是7,000M个，一个float16的参数占2个字节，所以需要`7,000M *2B=14GB`的显存，除了参数外，还要考虑存储KV缓存，以及样本在中间传输时的值，量化元信息(如果涉及量化的话)，所以需要留一些buffer。
2. 另外一个获取显存占用的方式是直接用这个[工具](https://github.com)\[1]，输入模型在huggingface上的名称，然后选择精度，就可以看到模型占用的显存大小了。


![model memory usage](https://nextcloud.aboydfd.com/s/QeeTswJASxpMrNR/preview)


需要注意的是，这里同样需要预留buffer，这上面的显存大小是纯模型本身的大小，量化的模型尤其要注意，需要考虑量化元数据带来的显存占用。


这样看下来，我这张12G显存的显卡，顶多只能跑一个7B\-int8的模型，为了能跑稍微大一点的上下文，我最终选择了Qwen/Qwen2\.5\-7B\-Instruct\-GPTQ\-Int4的模型（经过项目的实测，Qwen模型现在在中文开源领域确实很不错）。


兴奋地下载完的模型后，**噩梦**在启动vllm server的时候开始了。


迎面而来的是第一个错误是：


`RuntimeError: Error in model execution (input dumped to /tmp/err_execute_model_input_20241211-200011.pkl): CUDA error: no kernel image is available for execution on the device`


这个问题去stackoverflow\[2]了一下，大概率是vllm编译的时候没有支持对应的显卡架构，还记得重点么，没错，大概率就是不支持Pascal架构，我去官方文档\[3]上看了一下，确实没有发现Pascal的显卡支持，支持矩阵长这样，没有Pascal架构呀：


![vllm support matrix](https://nextcloud.aboydfd.com/s/A5ocHEHcJcxXkkk/preview)


没办法了，那就尝试自己编译vllm，看看能不能解决这个问题。


## 第二关：vllm编译


### 编译vllm


一开始编译的时候感觉还挺简单的，直接照着vllm的文档来，文档就只有一行命令`pip install -e .`，事情肯定没有这么简单，编译出错了：



```
CMake Error at CMakeLists.txt:252 (cuda_archs_loose_intersection):
        cuda_archs_loose_intersection Function invoked with incorrect arguments for
        function named: cuda_archs_loose_intersection

```

252行是这么写的：



```
cuda_archs_loose_intersection(MARLIN_ARCHS "8.0;8.6;8.7;8.9;9.0" ${CUDA_ARCHS})

```

中间经过了大量时间的定位，最终找到了问题所在，主要就是vllm设置了一个支持的显卡架构（其实它使用了算力来表示架构，算力和架构有对应关系\[4]）：



```
set(CUDA_SUPPORTED_ARCHS "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0")

```

只支持到7\.0算力，而Pascal架构是6\.1算力，所以最终CUDA\_ARCHS就为空，所以就报错了。


那简单呀，我直接给CUDA\_SUPPORTED\_ARCHS加上6\.1就行了，然后重新编译...


这次编译很顺利，编译完成之后，我就继续兴奋地启动vllm了，不幸的是，又一次报了`Cuda error: no kernel image is available for execution on the device`错误。


于是我继续Google，找到了这么一个github的issue\[5]，issue说这种情况是显卡不受支持了，需要自己编译（后面我自己测试了一下pytorch，其实我的pytorch是可以使用的，至于这里为什么报错，后续再研究研究吧），于是我就屁颠屁颠地去开始编译pytorch了。


### 编译pytorch


pytorch的编译就复杂很多了，不像vllm的编译命令，pytorch分了很多步。


先是要安装一堆前置工具：
\- CuDNN
\- cmake, ninja
\- requirements.txt
\- mkl\-static, mkl\-include
\- magma\-cuda121
\- triton



```
这些工具安装都还算顺利，要么照着说明安装，要么就是conda或pip安装，最后的triton就是一个make。

```

这里有几个坑：


1. pytorch要求先export CMAKE\_PREFIX\_PATH，并且给了个命令，检查一下执行完后的命令，有可能conda的路径没有找对，需要自己手动指定一下。
2. cmake一开始会找不到cudnn，需要将cudnn\-version.h（直接用find找一下自己安装的cudnn\-version.h在哪）文件拷贝或link到cuda的include目录下。


编译完成！


下面重新编译一次vllm，由于我们需要使用自己编译的pytorch，所以需要执行一下`python use_existing_torch.py`，vllm会帮我们把pytorch从依赖里删除掉，然后执行`pip install -r requirements-build.txt`，安装一下依赖，最后执行`pip install -e . --no-build-isolation`，这样安装的时候，vllm就不会再去安装这部分依赖了。


中间如果出现`version 'GLIBCXX_3.4.30' not found`的错误，我是把我安装的gcc的libstdc\+\+.so.6软链到conda的lib目录就行了。



```
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX_3.4.30

```

检查一下libstdc\+\+.so.6是否包含GLIBCXX\_3\.4\.30，如果包含，则软链到conda的lib目录下。



```
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ${CONDA_PREFIX_1}/lib/libstdc++.so.6

```

编译完成！


再次满怀期待地启动vllm server，不出意外地又报错了，这次报错是没找到xformers，这个是因为vllm默认是不带注意力后端的，因为它也不知道你用什么注意力后端，所以需要自己安装一下。安装的时候发现它依赖了pytorch并且去下载了pytorch，那要不还是自己编译一把吧。


xformers页面介绍中支持Pascal架构，所以安装起来很丝滑，一行命令即可：



```
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers

```

### 启动vllm server!


终于告一段落了，vllm server终于启动了，没有任何报错，我成功地看到了`Loading model weights took 5.2035 GB`。（这里可以印证我之前说的，量化的模型在考虑上量化元数据后，显存占用变大了很多，从计算得到的3\.5GB，变成了5\.2GB）


你以为故事到这就结束了？不不不，现在才是故事的开始。日志到了`Loading model weights took 5.2035 GB`就卡住了，我等了很久，发现它一直在卡在这。


## 第三关：定位性能问题的根因


### 初见端倪


出现这样的状况后，我是一点头绪都没有，只能像无头苍蝇一样，在vllm的Python代码里多打一些断点试试看了，在疯狂打了几十个断点之后，终于定位到卡哪了，vllm默认会先做一次profile run，来告诉你一些基本信息：



```
Memory profiling results: duration=11.82 seconds, total_gpu_memory=11.88GiB, initial_memory_usage=6.15GiB, peak_torch_memory=6.54GiB, memory_usage_post_profile=6.20GiB, non_torch_memory=1.05GiB, kv_cache_size=2.50GiB, gpu_memory_utilization=0.85.

```

因为这里需要进行模型推理，所以卡住了，这时候我才意识到，看一下nvidia\-smi看看显卡是否在工作其实就能知道它确实是在跑模型代码（虽然我一开始也有点意识到，却一直没往这个方面上想，毕竟再慢也不至于这么慢）。事实证明，卡住的时候，显卡确实在工作，所以问题很明显了，就是因为我的显卡推理速度“太慢”导致的。于是我就把`max-model-len`设置成了100，看看是否能够跑出结果来。等待了很长的时候后，服务真的启动了。


速度这么慢我是万万没有想到的，只能先换台机器测一下看怎么样，用了一台A6000的机器，发现人家一瞬间就启动了，那很明显了，问题就是只有我这边很慢。


### 初步定位问题


有了方向之后，那要做的事情就比较简单了，因为我自己编译了pytorch、xformers以及vllm，所以我需要一个个地排查。


先在pytorch官网上找到了跑[benchmark](https://github.com)\[6]的文档，分别在A6000机器、我的机器上自己编译的pytorch以及直接用pip install的pytorch上跑了一下，发现pytorch基础的性能是不差的。


然后使用xformers的[benchmark](https://github.com)\[7]，同样测试了一下，发现xformers的性能也是ok的。


那问题多半就出在vllm了，由于我不确定到底问题出在什么地方，以及我大概率确定基础库是没啥问题的，所以我打算把整个模型推理的各个步骤都记录一下执行时间，来看看具体是什么地方出问题了，按照28原则，问题大概率出在20%的地方。


接下来就是想办法记录时间了，我自己没有特别好的思路，所以就请教了一下LLM，LLM给了我一个思路，可以使用pytorch的`register_forward_pre_hook`和`register_forward_hook`来记录时间。它给的代码很粗糙直接使用time库来记录时间，而且只能记录一层模型。所以我就“稍”作修改，改成了递归地访问每一层模型，并且用cuda的`Event`（当然这个也是从LLM那问出来的）来记录时间。


时间记录的代码写完了，接下来就是运行一下，看看问题出在哪了。下面是我运行后跑出来的结果，各位来找找看觉得哪里有问题？



```
model: 134811.72338464856 ms
  model.embed_tokens: 37.62428665161133 ms
  model.layers: 134773.90933799744 ms
    model.layers.0: 4777.431374847889 ms
      model.layers.0.input_layernorm: 1.620192050933838 ms
      model.layers.0.self_attn: 673.7694255411625 ms
        model.layers.0.self_attn.qkv_proj: 411.43023681640625 ms
        model.layers.0.self_attn.rotary_emb: 0.1632319986820221 ms
        model.layers.0.self_attn.attn: 4.729087829589844 ms
        model.layers.0.self_attn.o_proj: 257.4468688964844 ms
      model.layers.0.post_attention_layernorm: 0.1900160014629364 ms
      model.layers.0.mlp: 4101.85174125433 ms
        model.layers.0.mlp.gate_up_proj: 2740.14697265625 ms
        model.layers.0.mlp.act_fn: 0.8391680121421814 ms
        model.layers.0.mlp.down_proj: 1360.8656005859375 ms

```

不得不说134s才跑完profile真的是离谱，然后确实就是28原则，问题就出在了4个地方，分别是：


* model.layers.0\.self\_attn.qkv\_proj
* model.layers.0\.self\_attn.o\_proj
* model.layers.0\.mlp.gate\_up\_proj
* model.layers.0\.mlp.down\_proj


这几个地方耗时都明显不正常，人家attention的计算才花了4ms，怎么这些操作要花几百甚至上千ms。


作为对比，我去查看了一下A6000机器上的结果：



```
model: 7459.573736906052 ms
  model.embed_tokens: 265.0838928222656 ms
  model.layers: 7192.4459400177 ms
    model.layers.0: 259.46213555336 ms
      model.layers.0.input_layernorm: 1.3496320247650146 ms
      model.layers.0.self_attn: 145.4847927093506 ms
        model.layers.0.self_attn.qkv_proj: 129.69778442382812 ms
        model.layers.0.self_attn.rotary_emb: 1.3486080169677734 ms
        model.layers.0.self_attn.attn: 3.180543899536133 ms
        model.layers.0.self_attn.o_proj: 11.257856369018555 ms
      model.layers.0.post_attention_layernorm: 2.0490241050720215 ms
      model.layers.0.mlp: 110.57868671417236 ms
        model.layers.0.mlp.gate_up_proj: 69.62483215332031 ms
        model.layers.0.mlp.act_fn: 4.104191780090332 ms
        model.layers.0.mlp.down_proj: 36.84966278076172 ms

```

结果很明显了，确实就是刚刚那几个地方的问题，其他地方的耗时基本上都差不多，有些甚至有领先（这个感觉应该属于误差）。


ok，知道问题了就去看看代码吧。


### 通过Python源码定位问题


经过一番研究，最终我把问题锁定到了量化计算上面，因为所有出问题的点都执行了量化的矩阵乘法计算。从网上搜了一张Qwen的架构图\[8]，我把耗时长的点都用红框标出来了。


![Qwen architecture](https://nextcloud.aboydfd.com/s/Fys66gZRTckSG36/preview)


从中我们可以看到，这些地方都执行了没有量化的输入和量化后的weight之间的矩阵乘法计算。


vllm的代码里则对应了：



```
class ColumnParallelLinear(LinearBase):
    ...
    def forward(self, input_):
        bias = self.bias if not self.skip_bias_add else None

        # Matrix multiply.
        assert self.quant_method is not None
        output_parallel = self.quant_method.apply(self, input_, bias) ## 就是这行进行了量化矩阵乘法
        if self.gather_output:
            # All-gather across the partitions.
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

```

和



```
class RowParallelLinear(LinearBase):
    ...
    def forward(self, input_):
        ...
        assert self.quant_method is not None
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        output_parallel = self.quant_method.apply(self,               ## 就是这行进行了量化矩阵乘法
                                                  input_parallel,
                                                  bias=bias_)
        ...

        return output, output_bias

```

由于我使用的是GPTQ量化模型，所以继续跟进需要去找的quant\_method是GPTQ相关的。
跟进到`self.quant_method.apply`:



```
class GPTQLinearMethod(LinearMethodBase):
    ...
    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        out_shape = x.shape[:-1] + (layer.qweight.shape[-1], )
        reshaped_x = x.reshape(-1, x.shape[-1])

        output = ops.gptq_gemm(reshaped_x, layer.qweight, layer.qzeros,
                               layer.scales, layer.g_idx,
                               layer.exllama_state == ExllamaState.READY,
                               self.quant_config.weight_bits)
        if bias is not None:
            output.add_(bias)
        return output.reshape(out_shape)


```

这里很明显问题就是gptq\_gemm的计算（GEMM表示General Matrix Multiplication，通用矩阵乘法），继续：



```
def gptq_gemm(a: torch.Tensor, b_q_weight: torch.Tensor,
              b_gptq_qzeros: torch.Tensor, b_gptq_scales: torch.Tensor,
              b_g_idx: torch.Tensor, use_exllama: bool,
              bit: int) -> torch.Tensor:
    return torch.ops._C.gptq_gemm(a, b_q_weight, b_gptq_qzeros, b_gptq_scales,
                                  b_g_idx, use_exllama, bit)

```

哎，最终还是得去看cuda代码么！！！


### 小插曲


这里想说一下GPTQ的名字，大家一看到可能会觉得它和GPT有关系，其实不是的，它这算是蹭GPT的热度，GPTQ的全称是Post\-Training Quantization for Generative pre\-trained transformers，确实是硬蹭的。Post\-Training Quantization，指的是训练后量化，所以它是一种在模型训练完之后，不再继续训练，单纯对权重和/或激活值进行量化的方法，而GPTQ是对PTQ的一种。


由于要去看cuda的源码，我对此没有很强的信心，我一没看过cuda源码，二不了解量化计算是什么样的，所以我就去紧急补课了一下，在网上找了个量化计算的视频\[9]来看，这个视频讲得很详细，对量化感兴趣的同学可以去看一下。看完视频过后我还不过瘾，我想弄清楚GPTQ的量化数学原理（GPTQ有一套完善的数学推理），只看了它的前身OBS、OBC、OBQ，在看GPTQ本身的时候，想到，我已经了解得足够多了，再看下去有点浪费时间了，还是回归主线先把。


感兴趣的同学可以参考下面2个链接，OBC/OBQ的论文本身写得也挺友好的，也可以看看：


1. [https://readpaper.feishu.cn/docx/OPP2dTuXAoaO0oxWhQAcC05Wnpc](https://github.com)
2. [https://zhuanlan.zhihu.com/p/646210009](https://github.com)
3. [https://arxiv.org/abs/2208\.11580](https://github.com)


### 通过cuda源码定位问题


接下来就是跟踪cuda源码了，通过搜索gptq\_gemm找到对应的cuda源码：



```
torch::Tensor gptq_gemm(torch::Tensor a, torch::Tensor b_q_weight,
                        torch::Tensor b_gptq_qzeros,
                        torch::Tensor b_gptq_scales, torch::Tensor b_g_idx,
                        bool use_exllama, int64_t bit) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto options = torch::TensorOptions().dtype(a.dtype()).device(a.device());
  at::Tensor c = torch::empty({a.size(0), b_q_weight.size(1)}, options);
  at::Tensor temp_dq = torch::empty(
      {b_q_weight.size(0) * 32 / bit, b_q_weight.size(1)}, options);

  vllm::gptq::gemm_half_q_half_cuda(
      at::cuda::getCurrentCUDABlasHandle(), (const half*)a.data_ptr(),
      (const uint32_t*)b_q_weight.data_ptr(),
      (const uint32_t*)b_gptq_qzeros.data_ptr(),
      (const half*)b_gptq_scales.data_ptr(),
      b_g_idx.device().is_meta() ? NULL : (const int*)b_g_idx.data_ptr(),
      (half*)c.data_ptr(), (half*)temp_dq.data_ptr(),
      c.size(0),              // m
      c.size(1),              // n
      a.size(1),              // k
      b_gptq_qzeros.size(0),  // group number
      use_exllama, bit);
  return c;
}


```

主要就是gemm\_half\_q\_half\_cuda这个函数，这个函数是GPTQ的量化矩阵乘法计算，a是输入，b\_q\_weight是量化后的权重，b\_gptq\_qzeros是公式里的Z，b\_gptq\_scales是公式里的S，然后use\_exllama是是否使用exllama库。


由于use\_exllama后续会影响到分支逻辑，所以先检查一下use\_exllama是否为true。从这里的代码一直往上翻查，可以看到use\_exllama是从config中读取的，qwen2\.5的config中设置的是true。


继续跟进代码：



```

void gemm_half_q_half_cuda(cublasHandle_t cublas_handle, const half* a,
                           const uint32_t* b_q_weight,
                           const uint32_t* b_gptq_qzeros,
                           const half* b_gptq_scales, const int* b_g_idx,
                           half* c, half* temp_dq, int size_m, int size_n,
                           int size_k, int groups, bool use_exllama, int bit) {
  bool use_reconstruct;
  if (use_exllama) {
    use_reconstruct = ((bit == 8 && size_m > MAX_Q_GEMM_ROWS_8BIT) ||
                       (bit != 8 && size_m > MAX_Q_GEMM_ROWS));
  } else {
    // The 2/3-bit kernels are somehow slower than dequant + gemm baseline, so
    // we disabled them for now.
    use_reconstruct = (bit < 4 || size_m > MAX_ALT_GEMM_ROWS);
  }
  if (use_reconstruct) {
    // Reconstruct FP16 matrix, then cuBLAS
    if (use_exllama) {
      reconstruct_exllama(b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx,
                          temp_dq, size_k, size_n, groups, bit);
    } else {
      reconstruct_gptq(b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx,
                       temp_dq, size_k, size_n, groups, bit);
    }

    const half alpha = __float2half(1.0f);
    const half beta = __float2half(0.0f);
    cublasHgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, size_n, size_m, size_k,
                &alpha, temp_dq, size_n, a, size_k, &beta, c, size_n);
} else if (use_exllama) {
    // Quantized matmul
    int max_chunks = size_m / BLOCK_M_SIZE_MAX;
    int last_chunk = max_chunks * BLOCK_M_SIZE_MAX;
    int last_chunk_size = size_m - last_chunk;

    if (max_chunks) {
      gemm_half_q_half_cuda_part(a, b_q_weight, b_gptq_qzeros, b_gptq_scales,
                                 b_g_idx, c, last_chunk, size_n, size_k,
                                 BLOCK_M_SIZE_MAX, groups, bit);
    }

    if (last_chunk_size) {
      gemm_half_q_half_cuda_part(a + last_chunk * size_k, b_q_weight,
                                 b_gptq_qzeros, b_gptq_scales, b_g_idx,
                                 c + last_chunk * size_n, last_chunk_size,
                                 size_n, size_k, last_chunk_size, groups, bit);
    }
  } else {
    gemm_half_q_half_alt(a, b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx,
                         c, size_m, size_n, size_k, bit);
  }

```

就是这部分代码，虽然现在看来比较明确它主要是走了use\_reconstruct\=True的分支，但是当时着实看了我很久的时间，要搞清楚走了哪个分支，得先知道这里的size\_m代表着什么，它其实表示着输入a的行数，也就是seq\_len\*batch\_size，而vllm在profile的时候，会使用到max\_token\_len大的seq\_len。


大部分应该都是大于MAX\_Q\_GEMM\_ROWS(\=50\)的，所以大部分是走了use\_reconstruct\=True的分支。这里我并没有深入研究reconstruct\_exllama和reconstruct\_gptq之间的差异点在哪，之后可以研究一下。


### 通过Nvidia的工具包定位问题


虽然代码大概看完了，但是我还是不知道到底是什么函数出问题了呀，那就只能用老法子了，要么打印，要么用profile工具。所以我就问了问GPT，它给我推荐了Nsight Compute，这是Nvidia出的一个工具，可以用来分析cuda程序的性能。吭哧吭哧学习了一下怎么用，然后现实给了我一顿暴击，Nsight Compute不支持Pascal架构，它的2019的版本才支持，但是2019的版本和现在的cuda版本又不兼容，尴尬。。。


不过幸运的是，在学习使用Nsight Compute的时候，我发现了Nsight System，这个也是Nvidia出的一个工具，可以用来分析cuda程序，看CPU和GPU联动的时候，问题出在哪，虽然不会像Nsight Compute那样详细地分析GPU的各个执行过程，但它能简单地分析cuda内核函数的耗时，这个正好是我现在需要的。


上结果：


![alt text](https://nextcloud.aboydfd.com/s/8NK7enYt4TqYntW/preview)


图中有两个关键信息：


1. 大部分的耗时都在2个内核函数上，就是`maxwell_hgemm_128x128`和`maxwell_hgemm_128x64`。
2. 在执行这俩函数前，都在执行reconstruct\_exllama内核函数。


这样的话就比较容易定位了，就是看reconstruct\_exllama后面执行了什么，那不就是`cublasHgemm`么。


### 和cublasHgemm较劲


经过一番搜索后，我了解了这个函数是啥，然后我就有点楞住了，啊？凭啥？这个是CuBLAS的函数，是Nvidia写的专门用来做向量和矩阵计算的，这怎么会有问题呢？这怎么能有问题呢？


为了验证它，我让GPT帮我写了个比较大的矩阵乘法并计算1000次来验证，结果确实是它的问题，执行起来很慢很慢，在A6000的机器上结果几乎是秒出，而我这边就会卡很久很久。


在这里我卡壳了好久，不知道这种情况下该咋办，感觉Pascal显卡就是该入土了，甚至想放弃了。后面想到，pytorch和xformers的性能不是没啥问题么，那肯定是有法子解决的。


于是我想了一个尝试的路子，我能不能换个库？我就去搜索了一下有没有CuBLAS的替代库。问了下GPT，还真就让我找到了，它就是CUTLASS，一个开源的CuBLAS库。


于是我就吭哧吭哧地又去编译了一下CUTLASS，3\.0版本开始的CUTLASS就不支持PASCAL了，所以我只能用2\.11版本。编译起来倒是异常丝滑，没有任何问题，和最新的cuda也能兼容。


编译完成后，我还是按照老思路，先找找看它的profile工具，确实有这个工具，于是我就进行了一次profile，就是CUTLASS的这次profile，帮我找到了问题的根因，官方的profile示例给的是用sgemm kernel:`./tools/profiler/cutlass_profiler --kernels=sgemm --m=4352 --n=4096 --k=4096`，我这边测试下来很快5s左右就执行完了，性能指标看着也还行：



```
Runtime: 15.7136  ms
Memory: 12.4296 GiB/s
Math: 9295.45 GFLOP/s

```

当时我并不知道sgemm kernel的s表示什么，但我猜到了和精度相关，我一开始还猜是small（其实它表示单精度single\-precision），就是精度很低，我就想，之前不是都是hgemm（半精度）么，我也来试试看它的profile是不是有这个kernel，这里纯属手贱，并不是想到了什么。但是就是这么一个意外，帮我找到了本次问题的根因。测试的结果是极其慢：



```
Runtime: 739.977  ms
Memory: 0.131972 GiB/s
Math: 197.391 GFLOP/s

```

我当时就在想，这差距也太大了吧，就算是small，也不应该small得这么厉害，能差这么多呀。我就又测了一下dgemm（双精度），结果和hgemm基本类似。


然后我就去确认了一下，sgemm表示的是单精度的运算。到这，我基本上能知道怎么回事了，大概率是Pascal架构不支持半精度的运算，导致计算效率很低。为了验证我这个想法，顺便作为学习，我去翻了Nvidia的官网，找了各个时期的架构白皮书，看了一下里面主要的显卡性能介绍。为了方便比较我先是让LLM帮我从各个白皮书里提取了性能信息，然后让它帮我输出json，我再用pandas将json转成了html方便我直观地对比。


这里给熟悉游戏显卡的同学稍微科普一下Nvidia的架构历史，从Maxwell开始：


* Maxwell 架构
	+ 发布时间：2014年
	+ 游戏卡命名：GTX 9xx 系列，如 GTX 970, GTX 980
	+ 数据卡命名：Tesla Mxx 系列，如 Tesla M40, Tesla M60
* Pascal 架构
	+ 发布时间：2016年
	+ 游戏卡命名：GTX 10xx 系列，如 GTX 1070, GTX 1080, GTX 1080 Ti
	+ 数据卡命名：Tesla Pxx 系列，如 Tesla P100
* Volta 架构
	+ 发布时间：2017年
	+ 游戏卡：N/A
	+ 数据卡命名：Tesla Vxx系列，如 Tesla V100
* Turing 架构
	+ 发布时间：2018年
	+ 游戏卡命名：RTX 20xx 系列，如 RTX 2070, RTX 2080, RTX 2080 Ti; GTX 16xx 系列如 GTX 1660, GTX 1660 Ti（不包含RT核的变体）
	+ 数据卡命名：Tesla Txx 系列，如 Tesla T4
* Ampere 架构
	+ 发布时间：2020年
	+ 游戏卡命名：RTX 30xx 系列，如 RTX 3070, RTX 3080, RTX 3090
	+ 数据卡命名：A100, A30
* Ada Lovelace 架构
	+ 发布时间：2022年
	+ 游戏卡命名：RTX 40xx 系列，如 RTX 4070, RTX 4080, RTX 4090
	+ 数据卡命名：L4
* Hopper 架构
	+ 发布时间：2022年
	+ 游戏卡命名：N/A
	+ 数据卡命名：H100
* Blackwell 架构
	+ 发布时间：2024年
	+ 游戏卡命名：N/A
	+ 数据卡命名：B100


![alt text](https://nextcloud.aboydfd.com/s/LYZbZz9qYJMFfjP/preview)


可以看到，Pascal架构的P100并没有fp16的支持， 而要有fp16支持的前提也是tensor core，Pascal架构是没有tensor core，只有cuda core的。然后也能发现，为什么说4090的推理性能能强过A100，因为它的各个算力指标都好于A100，A100强的是它显存大，显存带宽大，有SXM的支持，显卡之间的互联带宽高，所以在训练上有巨大的优势。


这下百分百确定问题所在了，没有fp16的支持，计算能力自然就很弱了。


## 第四关：优化性能


接下来就是改代码了，我的第一个想法是直接改成fp32的计算，这样计算速度就有保障了。但我还是决定去问一下LLM，看它有什么好的建议。它给我的建议是使用cublasGemmEx函数，这个函数也是CuBLAS的函数，它允许我们的输入输出矩阵都是fp16的，但是在计算的时候，转换成fp32来进行计算。


最后的改动就是这样：



```
    // cublasHgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, size_n, size_m, size_k,
    //             &alpha, temp_dq, size_n, a, size_k, &beta, c, size_n);
    cublasGemmEx(
      cublas_handle,                // Handle
      CUBLAS_OP_N,                  // transa
      CUBLAS_OP_N,                  // transb
      size_n,                       // m
      size_m,                       // n
      size_k,                       // k
      &alpha,                       // alpha
      temp_dq,                      // A
      CUDA_R_16F,                   // A type
      size_n,                       // lda
      a,                            // B
      CUDA_R_16F,                   // B type
      size_k,                       // ldb
      &beta,                        // beta
      c,                            // C
      CUDA_R_16F,                   // C type
      size_n,                       // ldc
      CUDA_R_32F,                   // computeType (FP32 for accumulation)
      CUBLAS_GEMM_DFALT_TENSOR_OP   // algo (default with potential Tensor Core usage)
    );

```

结果就如标题所说，这一行代码的更改，让性能提升了43倍，现在再来看一下我之前的pytorch的耗时日志：



```
model: 3098.3325251191854 ms
  model.embed_tokens: 33.70710372924805 ms
  model.layers: 3064.419405385852 ms
    model.layers.0: 131.46515500545502 ms
      model.layers.0.input_layernorm: 0.6445760130882263 ms
      model.layers.0.self_attn: 30.52022334933281 ms
        model.layers.0.self_attn.qkv_proj: 20.16111946105957 ms
        model.layers.0.self_attn.rotary_emb: 0.16473600268363953 ms
        model.layers.0.self_attn.attn: 3.8500161170959473 ms
        model.layers.0.self_attn.o_proj: 6.344351768493652 ms
      model.layers.0.post_attention_layernorm: 0.22275200486183167 ms
      model.layers.0.mlp: 100.07760363817215 ms
        model.layers.0.mlp.gate_up_proj: 65.92633819580078 ms
        model.layers.0.mlp.act_fn: 0.9378560185432434 ms
        model.layers.0.mlp.down_proj: 33.213409423828125 ms
    model.layers.1: 115.83395344018936 ms
      model.layers.1.self_attn: 17.98700802028179 ms
        model.layers.1.self_attn.rotary_emb: 0.16617600619792938 ms

```

可以看到，vllm的profile的耗时，从134s降到了3s，性能整整提升了43倍呀！！！


终于可以用我的Pascal显卡来推理了，爽！！


## 总结


### 第1点


对于一些程序员新人来说，希望这次的经历能给你一个参考，我们可以从一个问题点（一个好的问题从哪来确实也挺看运气的，我这次的问题刚好就是一个很深的问题，但是有时候我们可以刻意去创造一个问题，比如之前我看spark源码的时候，就是想搞清楚一个job的启动过程到底是怎么样的，这样也算是自己提出的一个好问题了）开始，然后一直深挖下去，这样你就熟悉了从表面一直到内核的整个过程，然后你就可以选择在任意感兴趣的地方开枝散叶，就能熟悉一整个框架乃至领域了。


对于我自己来说，我接下来能研究的就有：


* 再去研究一下GPTQ的量化过程，把数据原理完全搞懂，有机会的话自己可以跑一遍模型量化
* 看看GGUF的量化是怎么做的
* 看看GEMM具体是怎么计算的，有哪些点可以做来进行优化
* 去看看xformers的注意力计算是怎么做的
* 去看看vllm的kv cache是怎么做的
* 也可以去学学cuda编程
* ...


### 第2点


LLM在整个过程中起到了很大的作用，包括不限于：


1. 解释一些源码
2. 帮忙写部分测试用的代码
3. 帮忙澄清一些概念
4. 帮忙解释一些bug
5. ...


所以，赶紧用起来吧！


### 第3点


没事别瞎折腾别人不支持的东西，人家不支持是有原因的，除非你有折腾的觉悟和兴趣。


## 参考资料


1. [https://stackoverflow.com/questions/75682385/runtimeerror\-cuda\-error\-no\-kernel\-image\-is\-available\-for\-execution\-on\-the\-devi](https://github.com)
2. [https://docs.vllm.ai/en/latest/usage/compatibility\_matrix.html](https://github.com):[wgetcloud加速器官网下载](https://longdu.org)
3. [https://developer.nvidia.com/cuda\-gpus](https://github.com)
4. [https://github.com/pytorch/pytorch/issues/31285](https://github.com)
5. [https://pytorch.org/tutorials/recipes/recipes/benchmark.html](https://github.com)
6. [https://github.com/facebookresearch/xformers/blob/main/BENCHMARKS.md](https://github.com)
7. [https://blog.csdn.net/fan\_fan\_feng/article/details/138978901](https://github.com)
8. [https://www.bilibili.com/video/BV17m411f7Cm?spm\_id\_from\=333\.788\.videopod.sections\&vd\_source\=68452628e4137592ea9efa4793a102a6](https://github.com)


