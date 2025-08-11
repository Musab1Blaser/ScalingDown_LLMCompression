"""
Train config generator with external template files.
Generates YAML configs and bash scripts for training and merging.
"""

import argparse
import os
from pathlib import Path

def load_template(template_path):
    """Load template file from specified path."""
    with open(template_path, 'r') as f:
        return f.read()      
      
def generate_config(args, config_type, train_template, merge_template):
    """Generate configs for training and merging."""
    is_lora = config_type in ['lora', 'neo']
    has_teacher_model_name_or_path = config_type in ['kd', 'neo']
    
    # Handle LoRA configuration
    lora_config = ""
    if is_lora:
        lora_config = f"lora_rank: {args.lora_rank}\nlora_target: all"
        
    # Handle knowledge distillation configuration
    kd_config = ""
    if has_teacher_model_name_or_path:
        kd_config = f"teacher_model_name_or_path: {args.teacher_model_name_or_path}\nkd_ratio: {args.kd_ratio}"
    
    # Prepare training parameters
    output_dir = Path(args.output_root) / f"{args.model_family}-{args.model_size}" / ("lora" if is_lora else "full") / config_type
    train_params = {
        "model_name": args.base_model,
        "stage": "sft",
        "ft_type": "lora" if is_lora else "full",
        "lora_config": lora_config,
        "dataset": "alpaca_en",
        "template": args.template,
        "cutoff_len": args.cutoff_len,
        "max_samples": args.max_samples,
        "overwrite_cache": str(args.overwrite_cache).lower(),
        "output_dir": output_dir,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "plot_loss": str(args.plot_loss).lower(),
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": "{:.1e}".format(args.base_lr * (10 if is_lora else 1)),
        "epochs": args.epochs,
        "kd_config": kd_config
    }
    
    # Generate merge config for LoRA-based models
    merge_config = None
    if is_lora:
        merge_params = {
            "base_model": args.base_model,
            "adapter_path": output_dir,
            "template": args.template,
            "export_dir": f"output/{args.model_family}_{config_type}"
        }
        merge_config = merge_template.format(**merge_params)
    
    return train_template.format(**train_params), merge_config  # 修复此处
  
def main():
    # Load templates
    train_template_path = "examples/train_neo/llama3_template.yaml"
    merge_template_path = "examples/merge_lora/llama3_template.yaml"
    train_template = load_template(train_template_path)
    merge_template = load_template(merge_template_path)

    parser = argparse.ArgumentParser(description='Generate training configs')
    # 添加所有必要的参数
    parser.add_argument('--model_family', default='llama3', help='Base model family name')
    parser.add_argument('--model_size', default='3b', help='Model size variant')
    parser.add_argument('--base_model', default='AlphaAnas70/pruned_llama-3.2-3b-Instruct-20_per', help='HF model or local path')
    parser.add_argument('--teacher_model_name_or_path', default='meta-llama/Llama-3.2-3B-Instruct', help='Teacher model path can be empty or commented out')
    parser.add_argument('--base_lr', type=float, default=2e-5, help='Base learning rate')
    parser.add_argument('--epochs', type=float, default=1.0, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size per device')
    parser.add_argument('--grad_accum', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--cutoff_len', type=int, default=2048, help='Context length')
    parser.add_argument('--template', default='llama3', help='Prompt template')
    parser.add_argument('--max_samples', type=int, default=10000, help='Max samples per dataset')
    parser.add_argument('--lora_rank', type=int, default=64, help='LoRA rank')
    parser.add_argument('--kd_ratio', type=float, default=0.5, help='Knowledge distillation ratio')
    parser.add_argument('--logging_steps', type=int, default=10, help='Log interval')
    parser.add_argument('--save_steps', type=int, default=100, help='Save checkpoint interval')
    parser.add_argument('--plot_loss', action='store_true', help='Enable loss plotting')
    parser.add_argument('--overwrite_cache', action='store_true', help='Overwrite processed data')
    parser.add_argument('--output_root', default='saves', help='Root output directory')
    parser.add_argument('--run', default= ['neo'], nargs='*', choices=['sft', 'lora', 'kd', 'neo'], help='Configs to generate (default: all)')
    
    args = parser.parse_args()
    print(args)
    configs = args.run or ['sft', 'lora', 'kd', 'neo']
    
    # check for kd (kd/neo need teacher_model_name_or_path and kd_ratio>0)
    if not hasattr(args, 'teacher_model_name_or_path') or not args.teacher_model_name_or_path:
        args.teacher_model_name_or_path = ""
    if not (args.teacher_model_name_or_path and args.kd_ratio > 0):
        configs = [c for c in configs if c not in ('kd', 'neo')]
        if args.run in ('kd','neo'):
          print("\n\nWarning: For LoRA/Neo: --teacher_model_name_or_path should exits\n")

    # check for script
    if not configs:
        raise SystemExit("Error: No valid configurations to generate. Check parameters:\n"
                         "- For LoRA/Neo: --lora_rank must be >0\n"
                         "- For KD/Neo: --teacher_model_name_or_path and --kd_ratio>0 required")
        # 创建必要目录
    Path("examples/train_neo").mkdir(parents=True, exist_ok=True)
    Path("examples/merge_lora").mkdir(parents=True, exist_ok=True)
    
    bash_commands = []
    merge_commands = []
    output_dirs = []

    for config in configs:
        # 生成训练配置
        train_config, merge_config = generate_config(args, config, train_template, merge_template)
        train_yaml_path = f"examples/train_neo/{args.model_family}_{config}.yaml"
        with open(train_yaml_path, 'w') as f:
            f.write(train_config)
        bash_commands.append(f"llamafactory-cli train {train_yaml_path}")
        output_dirs.append(str(Path(args.output_root) / f"{args.model_family}-{args.model_size}" / ("lora" if config in ['lora', 'neo'] else "full") / config))

        # 生成合并配置
        if merge_config:
            merge_yaml_path = f"examples/merge_lora/{args.model_family}_{config}.yaml"
            with open(merge_yaml_path, 'w') as f:
                f.write(merge_config)
                
            merge_commands.append(f"llamafactory-cli export {merge_yaml_path}")

    # 生成执行脚本
    with open('run_train.sh', 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("#Training commands \n")
        f.write("\n".join(bash_commands))
        f.write("\n\n# Merging commands for LoRA/Neo only\n")
        f.write("\n".join(merge_commands))
        
    print("\n")
    print("Configuration files generated. Run:\n  bash run_train.sh")
    print("\nFinal model outputs will be saved to:")
    print("\n".join(f"- {d}" for d in output_dirs))
    print("\n")
          
if __name__ == "__main__":
    main()