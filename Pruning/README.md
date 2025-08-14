# Pruning Pipeline

This folder contains a pipeline for applying three pruning techniques (SparseGPT, Wanda and L1 Magnitude based Pruning). The pruning process is implemented in the [Pruning Script](final-pruning.ipynb) notebook.



## Models
The model used was Meta LLama : Llama-3.2-3B-Instruct for Pruning from which the following models are processed:
1. `Magnitude based L1 pruning at 17.56%`  
	[Link:https://huggingface.co/AlphaAnas70/pruned-llama-3b_torch_prune-l1_20_per]()
2. `SparseGPT pruning at 17.56%`  
	[Link: https://huggingface.co/AlphaAnas70/pruned_llama3b_sparsegpt_17.56_per]()
3. `DSnoT with SparseGPT pruning at 17.54%`  
	[Link: https://huggingface.co/AlphaAnas70/pruned_llama3b_dsnot_with_sparsegpt_17.54_per]()
4. `Wanda Pruning at 17.56%`  
	[Link: https://huggingface.co/AlphaAnas70/pruned_llama3b_Wanda-only_18_per]()
5. `DSnoT with Wanda Pruning at 17.56 %`  
	[Link:https://huggingface.co/AlphaAnas70/pruned_llama3b_DSnoT_with_Wanda_18_per]()


Refer to Github Link at [Github Repo](https://github.com/AlphaAnas/DSnoT) for the code implementation

Refer to the notebook for detailed implementation and results.



### References:
https://github.com/zyxxmu/DSnoT
https://github.com/VainF/Torch-Pruning/tree/master


