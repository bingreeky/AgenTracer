import argparse
import pathlib
import sys
import os

# Add the parent directory to sys.path to allow importing from coding package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import rich
from sandbox_fusion import RunStatus

from utils import code_exec

_EMPTY_RETURN_ = {
    "data_source": "kodcode",
    "task_id": None,
    "question": None,
    "test": None,
    "entry_point": None,
    "reference_solution": None,
}

block_libs = [
    "fake-useragent",
    "keras",
    "socket",
    "torch",
    "scipy",
    "sklearn",
    "cv2",
    "scipy",
    "imageio",
    "sphinx-pyproject",
    "xgboost",
    "tweepy",
    "flask",
    "matplotlib",
    "pillow",
    "seaborn",
    "smtpd",
    "pandas",
    "bs4",
]


def validate_single_example(example):
    """验证单个样本的代码，完全按照原始kodcode.py的逻辑"""
    reference_solution = example["reference_solution"]
    test_code = "from solution import *\n" + example["test"].strip()

    # 检查是否包含被阻止的库 - 这是唯一会返回_EMPTY_RETURN_的情况
    if any(lib in reference_solution for lib in block_libs):
        rich.print(
            f"[bold yellow]Skipping {example['task_id']} - contains blocked library"
        )
        return _EMPTY_RETURN_
    if any(lib in test_code for lib in block_libs):
        rich.print(
            f"[bold yellow]Skipping {example['task_id']} - test contains blocked library"
        )
        return _EMPTY_RETURN_

    # 执行代码验证 - 即使失败也返回完整样本（按照原始逻辑）
    try:
        result = code_exec(code=reference_solution, test=test_code)
        if result.status != RunStatus.Success:
            rich.print(f"[bold red]Test code failed for {example['task_id']}")
            rich.print("[bold red]Reference Solution:")
            print(reference_solution)
            rich.print("[bold red]Stdout:")
            print(result.run_result.stdout)
            rich.print("[bold red]Stderr:")
            print(result.run_result.stderr)
            # 按照原始逻辑，即使测试失败也返回完整样本
            rich.print(
                f"[bold yellow]⚠ {example['task_id']} - test failed but keeping sample"
            )
            return example
        else:
            rich.print(f"[bold green]✓ {example['task_id']} - validation passed")
            return example
    except Exception as e:
        rich.print(f"[bold red]Error validating {example['task_id']}: {str(e)}")
        # 按照原始逻辑，即使出现异常也返回完整样本
        rich.print(
            f"[bold yellow]⚠ {example['task_id']} - exception occurred but keeping sample"
        )
        return example


def validate_dataset(input_path, output_path, batch_size=100):
    """验证整个数据集"""
    rich.print(f"[bold blue]Loading dataset from {input_path}")

    # 读取parquet文件
    df = pd.read_parquet(input_path)
    rich.print(f"[bold green]Loaded {len(df)} samples")

    # 转换为字典列表
    examples = df.to_dict("records")

    # 分批验证
    validated_examples = []
    total_batches = (len(examples) + batch_size - 1) // batch_size

    for i in range(0, len(examples), batch_size):
        batch = examples[i : i + batch_size]
        batch_num = i // batch_size + 1

        rich.print(
            f"[bold blue]Processing batch {batch_num}/{total_batches} ({len(batch)} samples)"
        )

        for example in batch:
            validated_example = validate_single_example(example)
            if validated_example != _EMPTY_RETURN_:
                validated_examples.append(validated_example)

    # 保存验证后的数据
    if validated_examples:
        validated_df = pd.DataFrame(validated_examples)
        validated_df.to_parquet(output_path)
        rich.print(f"[bold green]Validation completed!")
        rich.print(f"[bold green]Original samples: {len(df)}")
        rich.print(f"[bold green]Validated samples: {len(validated_examples)}")
        rich.print(
            f"[bold green]Validation rate: {len(validated_examples)/len(df)*100:.2f}%"
        )
        rich.print(f"[bold green]Validated dataset saved to: {output_path}")
    else:
        rich.print(f"[bold red]No samples passed validation!")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Validate KodCode dataset with sandbox (following original logic)"
    )
    argparser.add_argument(
        "--input_path", required=True, help="Path to input parquet file"
    )
    argparser.add_argument(
        "--output_path",
        help="Path to output parquet file (auto-generated if not provided)",
    )
    argparser.add_argument(
        "--batch_size", type=int, default=100, help="Batch size for processing"
    )
    args = argparser.parse_args()

    # 设置输出路径
    input_path = pathlib.Path(args.input_path)
    if args.output_path is None:
        if input_path.is_dir():
            # 如果输入是文件夹，则为每个parquet文件生成一个输出文件名
            output_path = None  # 稍后在主逻辑中为每个文件单独生成
        else:
            output_path = input_path.parent / f"{input_path.stem}-validated.parquet"
    else:
        output_path = args.output_path

    # 确保输出目录存在（只有当output_path不为None时）
    if output_path is not None:
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    rich.print(
        "[bold blue]Starting KodCode dataset validation (following original logic)..."
    )
    rich.print(f"[bold blue]Input: {args.input_path}")
    rich.print(f"[bold blue]Batch size: {args.batch_size}")
    rich.print(
        "[bold yellow]Note: Following original logic - keeping samples even if tests fail"
    )

    if input_path.is_dir():
        # 处理目录输入：为每个parquet文件生成验证后的文件
        rich.print(f"[bold blue]Processing directory: {input_path}")
        parquet_files = list(input_path.glob("*.parquet"))
        if not parquet_files:
            rich.print(f"[bold red]No parquet files found in {input_path}")
            # 不能直接 return，因为在主模块中，应该用 sys.exit()
            import sys
            sys.exit(1)
        
        for parquet_file in parquet_files:
            rich.print(f"[bold blue]Processing file: {parquet_file}")
            output_file = parquet_file.parent / f"{parquet_file.stem}-validated.parquet"
            validate_dataset(str(parquet_file), str(output_file), args.batch_size)
    else:
        # 处理单个文件输入
        rich.print(f"[bold blue]Output: {output_path}")
        validate_dataset(args.input_path, output_path, args.batch_size)
