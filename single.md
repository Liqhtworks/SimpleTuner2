# SimpleTuner

transformer.single_transformer_blocks.0(3) 		

transformer.single_transformer_blocks.0.linear1(2) 		

transformer.single_transformer_blocks.0.linear1.lora_A.weight	[16, 3 072]	
BF16

transformer.single_transformer_blocks.0.linear1.lora_B.weight	[21 504, 16]	
BF16

transformer.single_transformer_blocks.0.norm.linear(2) 		

transformer.single_transformer_blocks.0.norm.linear.lora_A.weight	[16, 3 072]	
BF16

transformer.single_transformer_blocks.0.norm.linear.lora_B.weight	[9 216, 16]	
BF16

transformer.single_transformer_blocks.0.proj_out(2) 		

transformer.single_transformer_blocks.0.proj_out.lora_A.weight	[16, 15 360]	
BF16

transformer.single_transformer_blocks.0.proj_out.lora_B.weight	[3 072, 16]	
BF16

# fal

lora_unet_single_blocks_0(2) 		

lora_unet_single_blocks_0_linear(2) 		

lora_unet_single_blocks_0_linear1(2) 		

lora_unet_single_blocks_0_linear1.lora_down.weight	[16, 3 072]	
F32

lora_unet_single_blocks_0_linear1.lora_up.weight	[21 504, 16]	
F32

lora_unet_single_blocks_0_linear2(2) 		

lora_unet_single_blocks_0_linear2.lora_down.weight	[16, 15 360]	
F32

lora_unet_single_blocks_0_linear2.lora_up.weight	[3 072, 16]	
F32

lora_unet_single_blocks_0_modulation_lin(2) 		

lora_unet_single_blocks_0_modulation_lin.lora_down.weight	[16, 3 072]	
F32

lora_unet_single_blocks_0_modulation_lin.lora_up.weight	[9 216, 16]	
F32