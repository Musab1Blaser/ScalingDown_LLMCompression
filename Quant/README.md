# Quantization Pipeline

This folder contains a pipeline for applying three quantization techniques (BnB, AWQ, GPTQ) on 10 models. The quantization process is implemented in the `quant-pipeline.ipynb` notebook.

## Models
The following models are processed:
1. `musab1blaser/llama_3b_sparse_dsnot_neo2`
2. `musab1blaser/llama_3b_sparse_neo`
3. `musab1blaser/llama_3b_wanda_neo`
4. `musab1blaser/llama_3b_wanda_dsnot_neo`
5. `musab1blaser/llama_3b_pruned_neo`
6. `musab1blaser/llama_3b_sparse_dsnot_mini`
7. `musab1blaser/llama_3b_sparse_mini`
8. `musab1blaser/llama_3b_wanda_mini`
9. `musab1blaser/llama_3b_wanda_dsnot_mini`
10. `musab1blaser/llama_3b_pruned_mini`

## Quantization Techniques
### BnB (BitsAndBytes)
- **Config:**
  ```python
  BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.bfloat16,
      bnb_4bit_use_double_quant=True
  )
  ```

### AWQ (AutoAWQ)
- **Config:**
  ```python
  {
      "zero_point": True,
      "q_group_size": 128,
      "w_bit": 4,
      "version": "GEMM"
  }
  ```

### GPTQ
- **Config:**
  ```python
  GPTQConfig(
      bits=4,
      dataset=calibration_dataset,
      tokenizer=tokenizer
  )
  ```

Refer to the notebook for detailed implementation and results.