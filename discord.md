playerzer0x — 2:17 PM
hey ya'll - does anyone have a sample config for Kontext LoRA training? having trouble getting results with the same exact dataset that performs well when trained on fal. things i've tried:
steps between 1500 to 10k
standard PEFT lora
bs between 1-2, gradient accumulation between 1-4 on a H100
rank/alpha set to 16
learning rates between 5e-4 and 1e-5
optimi-lion, optimi-stableadamw, adamw_bf16
flow_schedule_auto_shift set to true and flow_schedule_shift set to 3
flux lora target with all+ffs, and others
fuse_qkv_projections set to true, with FA3 properly installed
 
validation outputs look like nothing is happening (adamw variants), or completely degrade into noise (optimi-lion)
here's my config in case anything stands out
{
  "--resume_from_checkpoint": "latest",
  "--data_backend_config": "config/multidatabackend.json",
  "--aspect_bucket_rounding": 2,
  "--seed": 42,
  "--minimum_image_size": 0,
Expand
config.json
3 KB
only things i can think of that i haven't tried are training at fp32, or increasing the GPU count, but don't think these are the issue
bghira 🐈
[STEP]
 — 2:23 PM
fuse_qkv_projections set to true, with FA3 properly installed

have you tried disabling this?
i'm not sure it works correctly
playerzer0x — 2:25 PM
will try, thank you
bghira 🐈
[STEP]
 — 2:26 PM
if i were to guess what caused kontext training to go fucky it would be this commit in the build_kontext_inputs method
GitHub
initial changes (to be tested tomorrow) · bghira/SimpleTuner@e2898c9
you can try reverting that logic to the old way it did things if you're using (presumably) single image inputs 
playerzer0x — 2:27 PM
can try throwing claude at it
playerzer0x — 2:27 PM
yep single
bghira 🐈
[STEP]
 — 2:28 PM
@yael i think had good results with this logic in place, but i've noticed in some experiments i was running with adding kontext editing to LibreFLUX and FluxBooru that things also stopped working as well there, hard to pinpoint exactly when it started being worse, but it seems to coincide with changes in how the IDs are made
