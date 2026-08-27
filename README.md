# AgenTracer Data Pipeline

![image](./assets/framework.png)



## 👋 Introduction

**AgenTracer** provides comprehensive tools for generating and evaluating training data that identifies **"Who"** (which agent/component) made mistakes and **"When"** (at which step) errors occurred in LLM agentic systems.

> **Data Release:** We have released an expanded version of the data [here](https://github.com/bingreeky/AgenTracer/releases/tag/data-v1.0.0). Due to differences in data versions and curation, the released data is not exactly identical to the version reported in the paper and contains additional samples.
>
> **Model Weights:** Due to internal considerations, we currently have no plans to release the AgenTracer-8B model weights.
>
> For any specific questions, please feel free to raise an issue or email me at guibinz@outlook.com, and I will do my best to help. Thank you!

This codebase provides a minimal working example where AgenTracer is applied to MetaGPT for data curation.

## 🌎 Setup

### Prerequisites

- Python 3.8+ (recommended: Python 3.11)
- Node.js and pnpm (required for MetaGPT)
- API keys for the services you plan to use



### Environment Setup

```bash
# Create MetaGPT environment
conda create -n metagpt python=3.11
conda activate metagpt

# Navigate to MetaGPT directory
cd MetaGPT

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Install Node.js dependencies
npm install -g pnpm

# Initialize MetaGPT configuration
metagpt --init-config
```

### 🔑 Configuration

#### MetaGPT Configuration

Edit `~/.metagpt/config2.yaml` with your API keys:
```yaml
llm:
  api_type: "openai"
  model: "gpt-4-turbo"
  base_url: "https://api.openai.com/v1"
  api_key: "YOUR_OPENAI_API_KEY"
```




## 🚀 Quick Start

### MetaGPT Framework

```bash
# Activate MetaGPT environment
conda activate metagpt

# Navigate to framework directory
cd MetaGPT/Who_When_Data_Pipeline/universal_framework

# Run with basic configuration
python universal_framework.py \
  --dataset kodcode \
  --work_dir /path/to/MetaGPT \
  --output /path/to/output \
  --max_rounds 3 \
  --max_tasks 10
```


## 🫡 Citation

If you find this repository helpful, a citation would be greatly appreciated:

```bibtex
@misc{zhang2025agentracer,
      title={AgenTracer: Who Is Inducing Failure in the LLM Agentic Systems?}, 
      author={Guibin Zhang and Junhao Wang and Junjie Chen and Wangchunshu Zhou and Kun Wang and Shuicheng Yan},
      year={2025},
      eprint={2509.03312},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.03312}, 
}
```
