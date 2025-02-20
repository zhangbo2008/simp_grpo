学习这个grop的算法实现.
核心在grpo_ref_split.py 这个代码. 注释直接写在这个代码里面.


test_qiudao.py : 进行detach的学习.






# 🚀🚀🚀 simple_GRPO 🚀🚀🚀
A very simple GRPO implement for reproducing r1-like LLM thinking.
This is a simple open source implementation that utilizes the core loss calculation formula referenced from Hugging Face's trl. 
We make the simplest codebase to support: 
- Save the GPU memory to make a feasible and efficient training. 
- Quickly understand RL processes such as GRPO from a teaching perspective. 
- Quickly try a lot of things, such as improved multi-answer generation, regrouping, penalty on KL, and parameter tuning.
- "Aha moment" is observed during the early stages of model training.

## 🌟 Features
### 💡 Simplicity
The project code is simple, with only about 200 lines of code spread across 2 files. It only depends on standard libraries such as _deepspeed_ and _torch_, without requiring dependencies like ray. It is designed to allow for more complex interventions.

### 🤖 Splited Reference Model
The reference model part is decoupled, which allows it to be run on different GPUs (even on a different machine with 4090). This avoids having the reference model and the training model on the same GPU, preventing multiple copies created by torch’s multiprocessing, and enabling training of a 7B model on 80G A800.

### 💃 Performance
Training completed in under 1 hour on 1*A800 GPUs. Both Qwen2.5-7B and Qwen2.5-3B exhibited an "Aha moment" within the first 30 optimization steps.

### 🥳 Core Loss Calculation
The loss calculation formula is based on Hugging Face's trl. We extend our gratitude to Hugging Face for their contribution.

## 🙌 Environment
The runtime environment is depicted below:
```
>> ds_report
torch version .................... 2.3.0+cu121
deepspeed info ................... 0.12.0
torch cuda version ............... 12.1
torch hip version ................ None
nvcc version ..................... 12.1
deepspeed wheel compiled w. ...... torch 2.3, cuda 12.1
shared memory (/dev/shm) size .... 1007.76 GB
```
At least two GPUs are needed.

## Usage

### JUST two py files, ref_server.py and grpo_ref_split.py are enough!
Run the following command:
``` bash
CUDA_VISIBLE_DEVICES=7 python ref_server.py
```
This just uses one GPU to collect and run the reference model.
We use http to transport data and logits.
They have so little data that they won't be any bottlenecks, http is the easiest to understand and has the fewest dependencies, and can be easily supported on multiple machines (we are using dozens of outdated 4090s to generate QA pairs for them!).

Then, open another bash:
``` bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 deepspeed grpo_ref_split.py
```
Use all other GPUs for training!

All parameters are in the code. We need to try more possibilities than a fking long argparse.

## ✨ Experimental Results

1. Runtime Environment
- Hardware Setup: 2×A800 (80GB) GPUs
- Configuration:
  - Training: 1 GPU with Zero-Stage 2 optimization
  - Inference: Dedicated 1 GPU (3090/4090 compatible)

2. Training Performance
   
| Model        | Steps | Time       |
|--------------|-------|------------|
| Qwen2.5-3B   | 60    | 12m 34s    |
| Qwen2.5-7B   | 60    | 16m 40s    |

2.1 Qwen2.5-3B

<img src="https://github.com/lsdefine/simple_GRPO/blob/main/images/Qwen2dot5-3B-res.jpg" alt="Qwen2.5-3B Accuracy" width="70%" style="display: inline-block; margin-right: 5px;">  

- Accuracy:
  - Stabilizes above 60% after 5 optimization steps
  - Peaks at ~70%
- Format Compliance:
  - Reaches ≈100% after 30 steps

2.2 Qwen2.5-7B

<img src="https://github.com/lsdefine/simple_GRPO/blob/main/images/Qwen2dot5-7B-res.jpg" alt="Qwen2.5-3B Accuracy" width="70%" style="display: inline-block; margin-right: 5px;">  

- Accuracy:
  - Maintains >90% throughout training
- Format Compliance:
  - Achieves 100% within 30 steps
- Convergence: Both metrics show rapid convergence

## Aha Moment
1. Qwen2.5-3B (Step 20)
> `<think>`
> ...
> To find out how many more eggs Camden painted than Arnold,  we subtract the number of eggs painted by Arnold from the number of eggs painted by Camden. So, 21 - 28 = -7.  **However, this result doesn't make sense in the context of the problem,
> as Camden cannot have painted fewer eggs than Arnold.
> Let's reconsider the initial solution steps:
> we should verify if the initial assumption about the relationship between Camden and Sarah's eggs is correct.**
> ...
> `</think>`
> `<answer>`-7`</answer>`

2. Qwen2.5-7B (Step 20)

> `<think>`
> ...
> Therefore, Joanne gathered 350 apples from the average trees.
> **However, this doesn't seem right because the total should be 500 and we've already accounted for 150,
> leaving room only for 350 from the average trees, which contradicts the total. Let's reassess.**
> ...
> `</think>`
> `<answer>`350`</answer>`

## ✨NEW
- 2025/02/19: Added a loss triton implementation, which has a little speedup, but you can choose not to use it.
- 2025/02/19: Added regroup version, implemented sampling of generated data on ref_server.

## 😊 TODO
- Answer generation may be invalid due to a group containing all wrong answers or all correct answers. We need group reorganization and better answer generation.
- GPU memory is still tight if it generates long cots. We have to split the groups to make the batch smaller.

We have implemented and are testing these features. They will be available soon.

## 🎉🎉🎉 Project Members

This project is led by Dr. Jiaqing Liang and Professor Yanghua Xiao from KnowledgeWorks Lab, Fudan University. The core development team includes Ph.D. candidate Jinyi Han, Master's student Xinyi Wang, and other contributors. We gratefully acknowledge their dedication to this work.

## 👏👏👏 Citation

If you find the code in our project useful, please consider citing our work as follows:

```
@misc{KW-R1,
  author = {Jiaqing Liang, Jinyi Han, Jie shi, Xinyi Wang, Weijia Li, Chengyuan Xiong, Tingyun Li, Yanghua Xiao},
  title = {KW-R1: A Simple Implementation of the GRPO Algorithm},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/lsdefine/simple_GRPO}},
}
```
"# simp_grpo" 
