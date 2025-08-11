burgalon — Yesterday at 9:15 AM
hey trying to catch up with recent changes
looking at trainer.py init_trainable_peft_adapter() it used to call
            target_modules = determine_adapter_target_modules(
                self.config, self.unet, self.transformer
            )


used to be
                transformer_lora_config = LoraConfig(
                    r=self.config.lora_rank,
                    lora_alpha=(
                        self.config.lora_alpha
                        if self.config.lora_alpha is not None
                        else self.config.lora_rank
                    ),
                    init_lora_weights=self.config.lora_initialisation_style,
                    target_modules=target_modules,
                    use_dora=self.config.use_dora,
                )


but now lora_target seems to be ignored

cc @bghira 🐈 
bghira 🐈
[STEP]
 — Yesterday at 9:27 AM
no it isn't ignored
            addkeys, misskeys = self.model.add_lora_adapter()
    def add_lora_adapter(self):
        target_modules = self.get_lora_target_layers()
        save_modules = self.get_lora_save_layers()
it's dynamic, and inside the Flux model class now
    def get_lora_target_layers(self):
        # Some models, eg. Flux should override this with more complex config-driven logic.
        if self.config.model_type == "lora" and (
            self.config.controlnet or self.config.control
        ):
            if "control" not in self.config.flux_lora_target.lower():
                logger.warning(
                    "ControlNet or Control is enabled, but the LoRA target does not include 'control'. Overriding to controlnet."
                )
            self.config.flux_lora_target = "controlnet"
        if self.config.lora_type.lower() == "standard":
            if self.config.flux_lora_target == "all":
                # target_modules = mmdit layers here
                return [
                    "to_k",
                    "to_q",
                    "to_v",
...
has to work, because ControlNet LoRA training wouldn't work without it
burgalon — Yesterday at 9:49 AM
but lora is initialized in helps/models/flux/pipeline.py  in
lora_config = LoraConfig(**lora_config_kwargs)
and does not include target modules
code goes here
    def init_trainable_peft_adapter(self):
        if "lora" not in self.config.model_type:
            return
into the return
bghira 🐈
[STEP]
 — Yesterday at 9:53 AM
if you're not training a lora, it will not be used
i think you're misunderstanding how it works
            lora_config_kwargs = get_peft_kwargs(
                rank, network_alpha_dict=network_alphas, peft_state_dict=state_dict
            )
it gets the lora config args from the state dict. that lora loader is used only when resuming training
the LoRA init happens inside helpers/models/common.py and helpers/models/flux/model.py 
burgalon — Yesterday at 9:56 AM
i'm using lora
bghira 🐈
[STEP]
 — Yesterday at 9:57 AM
then it wouldn't go into the return
let's start with why you think it's being ignored
because it is not being ignored
burgalon — Yesterday at 10:01 AM
because lora size is 36MB instead of 128MB after retrieving latest changes
using --model_type=lora --model_family=flux 
and after adding in flux/pipeline.py
            lora_config_kwargs['target_modules'] =determine_adapter_target_modules('[DESIRED_TARGET]', None, True)
            lora_config = LoraConfig(**lora_config_kwargs)

it works back again 
bghira 🐈
[STEP]
 — Yesterday at 10:03 AM
this determine_adapter_target_modules should not be used anywhere
it is dead code
that lora loader mixin is just copy-pasted in from Diffusers so that i can add ControlNet LoRA support to it
it is retrieving the target modules from the state dict which means something is going wrong earlier on
but you are looking in the wrong place(s)
burgalon — Yesterday at 10:11 AM
cmd/config arg lora_target is not taken into account in get_lora_target_layers in common.py? 
bghira 🐈
[STEP]
 — Yesterday at 10:12 AM
it's flux_lora_target
and the Flux model class overrides this
burgalon — Yesterday at 10:13 AM
it didn't use to override it... 
bghira 🐈
[STEP]
 — Yesterday at 10:13 AM
has for 4 months
since the refactor
burgalon — Yesterday at 10:14 AM
determine_adapter_target_modules basically should be removed?
bghira 🐈
[STEP]
 — Yesterday at 10:14 AM
yes
burgalon — Yesterday at 10:14 AM
is there a way to define target layers currently?
bghira 🐈
[STEP]
 — Yesterday at 10:14 AM
like custom list?
just the built in options, --flux_lora_target like all, all+ffs, context etc
burgalon — Yesterday at 10:15 AM
where is that taken into account?
bghira 🐈
[STEP]
 — Yesterday at 10:15 AM
i'm confused what is being asked
the common class has the abstracted function which uses the Flux lora target method to know how to create the LoraConfig
burgalon — Yesterday at 10:16 AM
ok i see
bghira 🐈
[STEP]
 — Yesterday at 10:16 AM
this is called in Trainer during init_trainable_peft_adapter
self.model.add_lora_adapter
burgalon — Yesterday at 10:17 AM
thank you
burgalon — Yesterday at 12:32 PM
is peft_model_precision also deprecated?
not seeing 
            logger.info(f"Moving LoRA adapter parameters to dtype {self.config.peft_model_precision}")
            for param in trainable_parameters:
                param.data = param.data.to(dtype=peft_dtype)
bghira 🐈
[STEP]
 — Yesterday at 1:06 PM
it is removed
i don't think it was a real user-exposed option
self.config is composed with more things as it initialises