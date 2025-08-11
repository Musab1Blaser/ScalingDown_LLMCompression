#!/bin/bash
#Training commands 
llamafactory-cli train examples/train_neo/llama3_sft.yaml
llamafactory-cli train examples/train_neo/llama3_lora.yaml
llamafactory-cli train examples/train_neo/llama3_kd.yaml
llamafactory-cli train examples/train_neo/llama3_neo.yaml

# Merging commands for LoRA/Neo only
llamafactory-cli export examples/merge_lora/llama3_lora.yaml
llamafactory-cli export examples/merge_lora/llama3_neo.yaml