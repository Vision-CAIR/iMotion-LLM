from transformers import Trainer
from trl.trainer.sft_config import SFTConfig # Import the custom config
from trl.trainer.sft_trainer import SFTTrainer  # Import the original SFTTrainer (assuming it's in `sft_trainer.py`)
from typing import Callable, Optional, Union
import dataclasses
import inspect
import os
import warnings
from typing import Callable, Optional, Union
import datasets
import torch
import torch.nn as nn
from accelerate.state import PartialState
from datasets import Dataset
from datasets.arrow_writer import SchemaInferenceError
from datasets.builder import DatasetGenerationError
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BaseImageProcessor,
    DataCollator,
    DataCollatorForLanguageModeling,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    is_wandb_available,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.utils import is_liger_kernel_available, is_peft_available
from transformers.utils.deprecation import deprecate_kwarg

from trl.extras.dataset_formatting import get_formatting_func_from_dataset
from trl.trainer.sft_config import SFTConfig
from trl.trainer.utils import (
    ConstantLengthDataset,
    DataCollatorForCompletionOnlyLM,
    # generate_model_card,
    # get_comet_experiment_url,
    peft_module_casting_to_bf16,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    SaveStrategy,
    TrainerMemoryTracker,
    TrainOutput,
    check_target_module_exists,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    neftune_post_forward_hook,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from transformers.utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    XLA_FSDPV2_MIN_VERSION,
    PushInProgress,
    PushToHubMixin,
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_galore_torch_available,
    is_grokadamw_available,
    is_in_notebook,
    is_ipex_available,
    is_liger_kernel_available,
    is_lomo_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_schedulefree_available,
    is_torch_compile_available,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_xla_available,
    is_torch_xpu_available,
    is_torchao_available,
    logging,
    strtobool,
)
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    ExportableState,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from packaging import version
import importlib.metadata
from transformers.trainer import _is_peft_model
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.integrations import (
    get_reporting_integration_callbacks,
    hp_params,
)
from transformers.utils.quantization_config import QuantizationMethod
from transformers.trainer import ALL_LAYERNORM_LAYERS, get_parameter_names, has_length

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if TYPE_CHECKING:
    import optuna

    if is_datasets_available():
        import datasets

logger = logging.get_logger(__name__)


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"

if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

if is_liger_kernel_available():
    from liger_kernel.transformers import AutoLigerKernelForCausalLM

if is_wandb_available():
    import wandb

from transformers import TrainerCallback

class CustomCallback(TrainerCallback):
    def __init__(self, custom_function):
        super().__init__()
        # self.custom_function = custom_function

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        """
        Executes a custom function before the optimizer step.
        """
        # Call the custom function with desired arguments
        # self.custom_function(args, state, control, **kwargs)
        pass

class CustomSFTTrainer(SFTTrainer):
    def __init__(
        self,
        model_: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: Optional[SFTConfig] = None,
        data_collator: Optional[DataCollator] = None,  # type: ignore
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional["PeftConfig"] = None,
        formatting_func: Optional[Callable] = None,
        non_llm_trainable_modules=None,
        pretraining=False,
        donot_prepare_model_for_kbit_training=False,
        scale_projection=False,
        scale_trajectory_modules_lr=1.0,
    ):
        self.scale_trajectory_modules_lr = scale_trajectory_modules_lr
        self.scale_projection=scale_projection
        self.pretraining = pretraining
        # self.non_llm_trainable_modules = non_llm_trainable_modules
        model = model_.llm
        if args is None:
            args = SFTConfig(output_dir="tmp_trainer")
        elif args is not None and args.__class__.__name__ == "TrainingArguments":
            args_as_dict = args.to_dict()
            # Manually copy token values as TrainingArguments.to_dict() redacts them
            args_as_dict.update({k: getattr(args, k) for k in args_as_dict.keys() if k.endswith("_token")})
            args = SFTConfig(**args_as_dict)

        if getattr(args, "model_init_kwargs", None) is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_init_kwargs to the SFTConfig, but your model is already instantiated.")
        else:
            model_init_kwargs = args.model_init_kwargs
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if torch_dtype is not None:
                # Convert to `torch.dtype` if an str is passed
                if isinstance(torch_dtype, str) and torch_dtype != "auto":
                    torch_dtype = getattr(torch, torch_dtype)
                if torch_dtype != "auto" and not isinstance(torch_dtype, torch.dtype):
                    raise ValueError(
                        f"Invalid `torch_dtype` passed to the SFTConfig. Expected a string with either `torch.dtype` or 'auto', but got {torch_dtype}."
                    )
                model_init_kwargs["torch_dtype"] = torch_dtype

        if isinstance(model, str):
            if args.use_liger:
                model = AutoLigerKernelForCausalLM.from_pretrained(model, **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if args.packing and data_collator is not None and isinstance(data_collator, DataCollatorForCompletionOnlyLM):
            raise ValueError(
                "You passed a `DataCollatorForCompletionOnlyLM` to the SFTTrainer. This is not compatible with the `packing` argument."
            )

        if is_peft_available() and peft_config is not None:
            if not isinstance(peft_config, PeftConfig):
                raise ValueError(
                    "If you want to use the PeftModel, you need to pass a PeftConfig object to the SFTTrainer."
                    f" and you passed a {type(peft_config)}."
                )

            if not isinstance(model, PeftModel):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )
                gradient_checkpointing_kwargs = getattr(args, "gradient_checkpointing_kwargs", None) or {}
                is_sharded_qlora = False
                # Below is to support QLoRA + FSDP / DS-Zero3 - one should never call
                # peft_module_casting_to_bf16 or prepare_model_for_kbit_training when doing
                # QLoRA + FSDP / DS-Zero3
                if getattr(model, "is_loaded_in_4bit", False):
                    for n, param in model.named_parameters():
                        if param.__class__.__name__ == "Params4bit":
                            is_sharded_qlora = param.data.device.type in {"cpu", "meta"}
                            break
                else:
                    is_sharded_qlora = False
                if getattr(model, "is_loaded_in_8bit", False) or (
                    getattr(model, "is_loaded_in_4bit", False) and not is_sharded_qlora
                ):
                    prepare_model_kwargs = {
                        "use_gradient_checkpointing": getattr(args, "gradient_checkpointing", False)
                    }

                    if _support_gc_kwargs:
                        prepare_model_kwargs["gradient_checkpointing_kwargs"] = gradient_checkpointing_kwargs

                    # model.model = prepare_model_for_kbit_training(model.model, **prepare_model_kwargs)
                    if not donot_prepare_model_for_kbit_training:
                        model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)
                    else:
                        ## Manual do all the things prepare_model_for_kbit_training except casting layer norms to float32
                        for name, param in model.named_parameters():
                            # print(name)
                            # freeze base model's layers
                            param.requires_grad = False

                    if args is not None:
                        args = dataclasses.replace(args, gradient_checkpointing=False)
                elif getattr(args, "gradient_checkpointing", False) and (
                    "use_reentrant" not in gradient_checkpointing_kwargs
                    or gradient_checkpointing_kwargs["use_reentrant"]
                ):
                    # For backward compatibility with older versions of transformers
                    if hasattr(model, "enable_input_require_grads"):
                        model.enable_input_require_grads()
                    else:

                        def make_inputs_require_grad(module, input, output):
                            output.requires_grad_(True)

                        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

                if (
                    "autocast_adapter_dtype" in list(inspect.signature(get_peft_model).parameters)
                    and getattr(model, "is_loaded_in_4bit", False)
                    and is_sharded_qlora
                ):
                    # model.model = get_peft_model(model.model, peft_config, autocast_adapter_dtype=False)
                    model = get_peft_model(model, peft_config, autocast_adapter_dtype=False)
                else:
                    # model.model = get_peft_model(model.model, peft_config)
                    model = get_peft_model(model, peft_config)
                if (
                    args is not None
                    and args.bf16
                    and (getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False))
                    and not is_sharded_qlora
                ):
                    if not donot_prepare_model_for_kbit_training:
                        peft_module_casting_to_bf16(model)
                    else:
                        for name, module in model.named_modules():
                            if any(x in name for x in ["lm_head", "embed_tokens"]):  # Specifically target these modules
                                if hasattr(module, "weight"):
                                    if module.weight.dtype != torch.float32:  # Only cast if not already float32
                                        module.to(torch.float32)
                            else:
                                if hasattr(module, "weight"):
                                    if module.weight.dtype == torch.float32:
                                        module = module.to(torch.bfloat16)

        if processing_class is None:
            processing_class = model_.llama_tokenizer
            # processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path)
            if getattr(processing_class, "pad_token", None) is None:
                processing_class.pad_token = processing_class.eos_token

        if args.max_seq_length is None:
            # to overcome some issues with broken tokenizers
            args.max_seq_length = min(processing_class.model_max_length, 1024)

        self.dataset_num_proc = args.dataset_num_proc
        self.dataset_batch_size = args.dataset_batch_size

        if args.dataset_kwargs is None:
            args.dataset_kwargs = {}

        if formatting_func is None:
            # check if dataset has ChatML format or instruction format and is supported
            # if not stays None
            formatting_func = get_formatting_func_from_dataset(train_dataset, processing_class)
            # if a template is detected, we don't need to add special tokens again
            if formatting_func is not None:
                args.dataset_kwargs["add_special_tokens"] = False

        if not args.packing:
            if data_collator is None:
                data_collator = DataCollatorForLanguageModeling(tokenizer=processing_class, mlm=False)

        # Pre-process the datasets only once per node. The remaining processes will use the cache.
        with PartialState().local_main_process_first():
            if train_dataset is not None:
                train_dataset = self._prepare_dataset(
                    train_dataset,
                    processing_class,
                    args.packing,
                    args.dataset_text_field,
                    args.max_seq_length,
                    formatting_func,
                    args.num_of_sequences,
                    args.chars_per_token,
                    remove_unused_columns=args.remove_unused_columns if args is not None else True,
                    **args.dataset_kwargs,
                )
            if eval_dataset is not None:
                _multiple = isinstance(eval_dataset, dict)
                _eval_datasets = eval_dataset if _multiple else {"singleton": eval_dataset}

                eval_packing = args.packing if args.eval_packing is None else args.eval_packing

                for _eval_dataset_name, _eval_dataset in _eval_datasets.items():
                    _eval_datasets[_eval_dataset_name] = self._prepare_dataset(
                        _eval_dataset,
                        processing_class,
                        eval_packing,
                        args.dataset_text_field,
                        args.max_seq_length,
                        formatting_func,
                        args.num_of_sequences,
                        args.chars_per_token,
                        remove_unused_columns=args.remove_unused_columns if args is not None else True,
                        **args.dataset_kwargs,
                    )
                if not _multiple:
                    eval_dataset = _eval_datasets["singleton"]

        if processing_class.padding_side is not None and processing_class.padding_side != "right":
            warnings.warn(
                "You passed a processing_class with `padding_side` not equal to `right` to the SFTTrainer. This might "
                "lead to some unexpected behaviour due to overflow issues when training a model in half-precision. "
                "You might consider adding `processing_class.padding_side = 'right'` to your code.",
                UserWarning,
            )
        
        model_.llm = model
        self.custom_super_init(
            model_=model_,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        # super().__init__(
        #     model=model,
        #     args=args,
        #     data_collator=data_collator,
        #     train_dataset=train_dataset,
        #     eval_dataset=eval_dataset,
        #     processing_class=processing_class,
        #     model_init=model_init,
        #     compute_metrics=compute_metrics,
        #     callbacks=callbacks,
        #     optimizers=optimizers,
        #     preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        # )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)
        if hasattr(self.model.llm, "add_model_tags"):
            self.model.llm.add_model_tags(self._tag_names)

        if self.train_dataset is not None:
            if self.args.max_steps > 0 and args.packing:
                self.train_dataset.infinite = True
            elif self.args.max_steps == -1 and args.packing:
                self.train_dataset.infinite = False
    
    def custom_super_init(
        self,
        model_: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset, "datasets.Dataset"]] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset], "datasets.Dataset"]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_loss_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        optimizer_cls_and_kwargs: Optional[Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]] = None,
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        # model = model_.llm
        model = model_
        if args is None:
            output_dir = "tmp_trainer"
            logger.info(f"No `TrainingArguments` passed, using `output_dir={output_dir}`.")
            args = TrainingArguments(output_dir=output_dir)
        if args.batch_eval_metrics and compute_metrics is not None:
            if "compute_result" not in inspect.signature(compute_metrics).parameters.keys():
                raise ValueError(
                    "When using `batch_eval_metrics`, your `compute_metrics` function must take a `compute_result`"
                    " boolean argument which will be triggered after the last batch of the eval set to signal that the"
                    " summary statistics should be returned by the function."
                )
        if args.eval_strategy is not None and args.eval_strategy != "no" and eval_dataset is None:
            raise ValueError(
                f"You have set `args.eval_strategy` to {args.eval_strategy} but you didn't pass an `eval_dataset` to `Trainer`. Either set `args.eval_strategy` to `no` or pass an `eval_dataset`. "
            )
        if args.save_strategy == SaveStrategy.BEST or args.load_best_model_at_end:
            if args.metric_for_best_model is None:
                raise ValueError(
                    "`args.metric_for_best_model` must be provided when using 'best' save_strategy or if `args.load_best_model_at_end` is set to `True`."
                )

        self.args = args
        self.compute_loss_func = compute_loss_func
        # Seed must be set before instantiating the model when using model
        enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)

        self.hp_name = None
        self.deepspeed = None
        self.is_in_train = False

        self.create_accelerator_and_postprocess()

        # memory metrics - must set up as early as possible
        self._memory_tracker = TrainerMemoryTracker(self.args.skip_memory_metrics)
        self._memory_tracker.start()

        # set the correct log level depending on the node
        log_level = args.get_process_log_level()
        logging.set_verbosity(log_level)

        # force device and distributed setup init explicitly
        args._setup_devices

        if model is None:
            if model_init is not None:
                self.model_init = model_init
                model = self.call_model_init()
            else:
                raise RuntimeError("`Trainer` requires either a `model` or `model_init` argument")
        else:
            if model_init is not None:
                warnings.warn(
                    "`Trainer` requires either a `model` or `model_init` argument, but not both. `model_init` will"
                    " overwrite your model when calling the `train` method. This will become a fatal error in the next"
                    " release.",
                    FutureWarning,
                )
            self.model_init = model_init

        if model.__class__.__name__ in MODEL_MAPPING_NAMES:
            raise ValueError(
                f"The model you have picked ({model.__class__.__name__}) cannot be used as is for training: it only "
                "computes hidden states and does not accept any labels. You should choose a model with a head "
                "suitable for your task like any of the `AutoModelForXxx` listed at "
                "https://huggingface.co/docs/transformers/model_doc/auto"
            )

        if getattr(model, "is_parallelizable", False) and getattr(model, "model_parallel", False):
            self.is_model_parallel = True
        else:
            self.is_model_parallel = False

        if getattr(model.llm, "hf_device_map", None) is not None:
            devices = [device for device in set(model.llm.hf_device_map.values()) if device not in ["cpu", "disk"]]
            if len(devices) > 1:
                self.is_model_parallel = True
            elif len(devices) == 1:
                self.is_model_parallel = self.args.device != torch.device(devices[0])
            else:
                self.is_model_parallel = False

            # warn users
            if self.is_model_parallel:
                logger.info(
                    "You have loaded a model on multiple GPUs. `is_model_parallel` attribute will be force-set"
                    " to `True` to avoid any unexpected behavior such as device placement mismatching."
                )

        if self.args.use_liger_kernel:
            if is_liger_kernel_available():
                from liger_kernel.transformers import _apply_liger_kernel_to_instance

                if isinstance(model, PreTrainedModel):
                    # Patch the model with liger kernels. Use the default kernel configurations.
                    _apply_liger_kernel_to_instance(model=model)
                else:
                    logger.warning(
                        "The model is not an instance of PreTrainedModel. No liger kernels will be applied."
                    )
            else:
                raise ImportError(
                    "You have set `use_liger_kernel` to `True` but liger-kernel >= 0.3.0 is not available. "
                    "Please install it with `pip install liger-kernel`"
                )
        model_ = model
        model = model_.llm
        _is_quantized_and_base_model = getattr(model, "is_quantized", False) and not getattr(
            model, "_hf_peft_config_loaded", False
        )
        _quantization_method_supports_training = (
            getattr(model, "hf_quantizer", None) is not None and model.hf_quantizer.is_trainable
        )

        _is_model_quantized_and_qat_trainable = getattr(model, "hf_quantizer", None) is not None and getattr(
            model.hf_quantizer, "is_qat_trainable", False
        )

        # Filter out quantized + compiled models
        if _is_quantized_and_base_model and hasattr(model, "_orig_mod"):
            raise ValueError(
                "You cannot fine-tune quantized model with `torch.compile()` make sure to pass a non-compiled model when fine-tuning a quantized model with PEFT"
            )

        # At this stage the model is already loaded
        if _is_quantized_and_base_model and not _is_peft_model(model) and not _is_model_quantized_and_qat_trainable:
        # if _is_quantized_and_base_model and not _is_peft_model(model.model) and not _is_model_quantized_and_qat_trainable:
            raise ValueError(
                "You cannot perform fine-tuning on purely quantized models. Please attach trainable adapters on top of"
                " the quantized model to correctly perform fine-tuning. Please see: https://huggingface.co/docs/transformers/peft"
                " for more details"
            )
        elif _is_quantized_and_base_model and not _quantization_method_supports_training:
            raise ValueError(
                f"The model you are trying to fine-tune is quantized with {model.hf_quantizer.quantization_config.quant_method}"
                " but that quantization method do not support training. Please open an issue on GitHub: https://github.com/huggingface/transformers"
                f" to request the support for training support for {model.hf_quantizer.quantization_config.quant_method}"
            )

        self.is_fsdp_xla_enabled = args.fsdp_config["xla"]
        if len(args.fsdp) > 0:
            if self.is_deepspeed_enabled:
                raise ValueError(
                    "Using --fsdp xxx together with --deepspeed is not possible, deactivate one of those flags."
                )
            if not args.fsdp_config["xla"] and args.parallel_mode != ParallelMode.DISTRIBUTED:
                raise ValueError("Using fsdp only works in distributed training.")

        # one place to sort out whether to place the model on device or not
        # postpone switching model to cuda when:
        # 1. MP - since we are trying to fit a much bigger than 1 gpu model
        # 2. fp16-enabled DeepSpeed loads the model in half the size and it doesn't need .to() anyway,
        #    and we only use deepspeed for training at the moment
        # 3. full bf16 or fp16 eval - since the model needs to be cast to the right dtype first
        # 4. FSDP - same as MP
        self.place_model_on_device = args.place_model_on_device
        if (
            self.is_model_parallel
            or self.is_deepspeed_enabled
            or ((args.fp16_full_eval or args.bf16_full_eval) and not args.do_train)
            or self.is_fsdp_xla_enabled
            or self.is_fsdp_enabled
        ):
            self.place_model_on_device = False

        default_collator = (
            DataCollatorWithPadding(processing_class)
            if processing_class is not None
            and isinstance(processing_class, (PreTrainedTokenizerBase, SequenceFeatureExtractor))
            else default_data_collator
        )
        self.data_collator = data_collator if data_collator is not None else default_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.processing_class = processing_class
        model_.llm = model
        # Bnb Quantized models doesn't support `.to` operation.
        # if (
        #     self.place_model_on_device
        #     and not getattr(model, "quantization_method", None) == QuantizationMethod.BITS_AND_BYTES
        # ):
        #     self._move_model_to_device(model, args.device)
        if (
            self.place_model_on_device
            and not getattr(model_, "quantization_method", None) == QuantizationMethod.BITS_AND_BYTES
        ):
            self._move_model_to_device(model_, args.device)

        # Force n_gpu to 1 to avoid DataParallel as MP will manage the GPUs
        if self.is_model_parallel:
            self.args._n_gpu = 1

        # later use `self.model is self.model_wrapped` to check if it's wrapped or not
        # model_.llm = model
        self.model_wrapped = model_
        self.model = model_

        # Just in case the model was wrapped outside of the `Trainer`
        unwrapped_model = self.accelerator.unwrap_model(model_)
        # unwrapped_model = self.accelerator.unwrap_model(model)
        model_forward = (
            unwrapped_model.forward
            if not _is_peft_model(unwrapped_model)
            else unwrapped_model.get_base_model().forward
        )
        forward_params = inspect.signature(model_forward).parameters
        self.model_accepts_loss_kwargs = any(k.kind == inspect.Parameter.VAR_KEYWORD for k in forward_params.values())
        self.neftune_noise_alpha = args.neftune_noise_alpha

        self.compute_metrics = compute_metrics
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics
        self.optimizer, self.lr_scheduler = optimizers
        self.optimizer_cls_and_kwargs = optimizer_cls_and_kwargs
        if self.optimizer_cls_and_kwargs is not None and self.optimizer is not None:
            raise RuntimeError("Passing both `optimizers` and `optimizer_cls_and_kwargs` arguments is incompatible.")
        if model_init is not None and (self.optimizer is not None or self.lr_scheduler is not None):
            raise RuntimeError(
                "Passing a `model_init` is incompatible with providing the `optimizers` argument. "
                "You should subclass `Trainer` and override the `create_optimizer_and_scheduler` method."
            )
        if is_torch_xla_available() and self.optimizer is not None:
            for param in self.model.parameters():
                model_device = param.device
                break
            for param_group in self.optimizer.param_groups:
                if len(param_group["params"]) > 0:
                    optimizer_device = param_group["params"][0].device
                    break
            if model_device != optimizer_device:
                raise ValueError(
                    "The model and the optimizer parameters are not on the same device, which probably means you"
                    " created an optimizer around your model **before** putting on the device and passing it to the"
                    " `Trainer`. Make sure the lines `import torch_xla.core.xla_model as xm` and"
                    " `model.to(xm.xla_device())` is performed before the optimizer creation in your script."
                )
        if (self.is_fsdp_xla_enabled or self.is_fsdp_enabled) and (
            self.optimizer is not None or self.lr_scheduler is not None
        ):
            raise RuntimeError(
                "Passing `optimizers` is not allowed if PyTorch FSDP is enabled. "
                "You should subclass `Trainer` and override the `create_optimizer_and_scheduler` method."
            )
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.model, self.processing_class, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)

        # Will be set to True by `self._setup_loggers()` on first call to `self.log()`.
        self._loggers_initialized = False

        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

        if not callable(self.data_collator) and callable(getattr(self.data_collator, "collate_batch", None)):
            raise ValueError("The `data_collator` should be a simple callable (function, class with `__call__`).")

        if args.max_steps > 0 and args.num_train_epochs > 0:
            logger.info("max_steps is given, it will override any value given in num_train_epochs")

        if train_dataset is not None and not has_length(train_dataset) and args.max_steps <= 0:
            raise ValueError(
                "The train_dataset does not implement __len__, max_steps has to be specified. "
                "The number of steps needs to be known in advance for the learning rate scheduler."
            )

        if (
            train_dataset is not None
            and isinstance(train_dataset, torch.utils.data.IterableDataset)
            and args.group_by_length
        ):
            raise ValueError("the `--group_by_length` option is only available for `Dataset`, not `IterableDataset")

        self._signature_columns = None

        # Mixed precision setup
        self.use_apex = False
        self.use_cpu_amp = False

        # Mixed precision setup for SageMaker Model Parallel
        if is_sagemaker_mp_enabled():
            # BF16 + model parallelism in SageMaker: currently not supported, raise an error
            if args.bf16:
                raise ValueError("SageMaker Model Parallelism does not support BF16 yet. Please use FP16 instead ")

            if IS_SAGEMAKER_MP_POST_1_10:
                # When there's mismatch between SMP config and trainer argument, use SMP config as truth
                if args.fp16 != smp.state.cfg.fp16:
                    logger.warning(
                        f"FP16 provided in SM_HP_MP_PARAMETERS is {smp.state.cfg.fp16}, "
                        f"but FP16 provided in trainer argument is {args.fp16}, "
                        f"setting to {smp.state.cfg.fp16}"
                    )
                    args.fp16 = smp.state.cfg.fp16
            else:
                # smp < 1.10 does not support fp16 in trainer.
                if hasattr(smp.state.cfg, "fp16"):
                    logger.warning(
                        f"FP16 provided in SM_HP_MP_PARAMETERS is {smp.state.cfg.fp16}, "
                        "but SageMaker Model Parallelism < 1.10 does not support FP16 in trainer."
                    )
        if (args.fp16 or args.bf16) and args.half_precision_backend == "auto":
            if args.device == torch.device("cpu"):
                if args.fp16:
                    if not is_torch_greater_or_equal_than_2_3:
                        raise ValueError("Tried to use `fp16` but it is not supported on cpu")
                else:
                    args.half_precision_backend = "cpu_amp"
            logger.info(f"Using {args.half_precision_backend} half precision backend")

        if (args.fp16 or args.bf16) and not (self.is_deepspeed_enabled or is_sagemaker_mp_enabled()):
            # deepspeed and SageMaker Model Parallel manage their own half precision
            if args.half_precision_backend == "cpu_amp":
                self.use_cpu_amp = True
                self.amp_dtype = torch.bfloat16
            elif args.half_precision_backend == "apex":
                if not is_apex_available():
                    raise ImportError(
                        "Using FP16 with APEX but APEX is not installed, please refer to"
                        " https://www.github.com/nvidia/apex."
                    )
                self.use_apex = True

        # Label smoothing
        if self.args.label_smoothing_factor != 0:
            self.label_smoother = LabelSmoother(epsilon=self.args.label_smoothing_factor)
        else:
            self.label_smoother = None

        self.control = TrainerControl()

        self.state = TrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ],
        )
        # Internal variable to count flos in each process, will be accumulated in `self.state.total_flos` then
        # returned to 0 every time flos need to be logged
        self.current_flos = 0
        self.hp_search_backend = None
        default_label_names = find_labels(self.model.__class__)
        self.label_names = default_label_names if self.args.label_names is None else self.args.label_names
        self.can_return_loss = can_return_loss(self.model.__class__)
        self.control = self.callback_handler.on_init_end(self.args, self.state, self.control)

        # Internal variables to help with automatic batch size reduction
        self._train_batch_size = args.train_batch_size
        self._created_lr_scheduler = False

        # very last
        self._memory_tracker.stop_and_update_metrics()

        # torch.compile
        if args.torch_compile and not is_torch_compile_available():
            raise RuntimeError("Using torch.compile requires PyTorch 2.0 or higher.")

        self.is_fsdp_xla_v2_enabled = args.fsdp_config.get("xla_fsdp_v2", False)
        if self.is_fsdp_xla_v2_enabled:
            if not IS_XLA_FSDPV2_POST_2_2:
                raise ValueError("FSDPv2 requires `torch_xla` 2.2 or higher.")
            # Prepare the SPMD mesh that is going to be used by the data loader and the FSDPv2 wrapper.
            # Tensor axis is just a placeholder where it will not be used in FSDPv2.
            num_devices = xr.global_runtime_device_count()
            xs.set_global_mesh(xs.Mesh(np.array(range(num_devices)), (num_devices, 1), axis_names=("fsdp", "tensor")))
        self.is_fsdp_xla_v1_enabled = self.is_fsdp_xla_enabled and not self.is_fsdp_xla_v2_enabled

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        # outputs = model(**inputs)
        outputs = model(inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss
    
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        # pyre-fixme[16]: `Trainer` has no attribute `model`.
        opt_model = self.model
        # if self.args.unfreeze_mm_vision_tower:
        #     opt_model.get_model().vision_tower_aux_list = nn.ModuleList(opt_model.get_vision_tower_aux_list())
        #     self.param_to_name = map_params_to_module_names([opt_model])
        # pyre-fixme[16]: `Trainer` has no attribute `optimizer`.
        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model.llm, ALL_LAYERNORM_LAYERS)
            decay_parameters = [f"llm.{name}" for name in decay_parameters if "bias" not in name]
            # decay_parameters = self.get_decay_parameter_names(opt_model)
            
            if True:
                all_modules_names = [n for n,_ in opt_model.named_parameters()]
                non_llm_trainable_modules_names = [
                    name for name in all_modules_names if name[:4]!='llm.'
                ]
                ## non-llm custom projection modules
                input_projection_non_llm_trainable_modules_names = [n for n in non_llm_trainable_modules_names if 'llm_input_adapter' in n]
                output_projection_non_llm_trainable_modules_names = [n for n in non_llm_trainable_modules_names if 'llm_output_adapter' in n]
                projection_modules_names = input_projection_non_llm_trainable_modules_names+output_projection_non_llm_trainable_modules_names
                ## non-llm custom backbone encoder and decoder modules
                other_non_llm_trainable_modules_names = [n for n in non_llm_trainable_modules_names if n not in projection_modules_names]
                projection_params = []
                custom_modules_params = []
                llm_decay_params = []
                llm_no_decay_params = []
                for n, p in opt_model.named_parameters():
                    if n in projection_modules_names and p.requires_grad:
                        # pass
                        projection_params.append(p)
                    elif n in other_non_llm_trainable_modules_names and p.requires_grad:
                        # print(n)
                        # pass
                        custom_modules_params.append(p)
                    elif p.requires_grad: # llm parameters that requires grad
                        # pass
                        if n in decay_parameters:
                            llm_decay_params.append(p)
                        else:   
                            llm_no_decay_params.append(p)
                
                optimizer_grouped_parameters = [
                    {
                        "params": projection_params,
                        "weight_decay": 0.0,
                        # "weight_decay": self.args.weight_decay,
                        "lr": self.args.learning_rate if not self.scale_projection else self.args.learning_rate*10,
                        # "lr": self.args.learning_rate*10,
                        # "lr": self.args.learning_rate,
                    },
                    {
                        "params": custom_modules_params,
                        "weight_decay": 0.0,
                        # "weight_decay": self.args.weight_decay,
                        # "lr": self.args.learning_rate*0.1,
                        # "lr": self.args.learning_rate*10,
                        # "lr": self.args.learning_rate,
                        "lr": self.args.learning_rate*self.scale_trajectory_modules_lr,
                    },
                    {
                        "params": llm_decay_params,
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.learning_rate,
                    },
                    {
                        "params": llm_no_decay_params,
                        "weight_decay": 0.0,
                        # "weight_decay": self.args.weight_decay,
                        "lr": self.args.learning_rate,
                    },
                ]
            elif len(self.non_llm_trainable_modules)>0:
                all_modules_names = [n for n,_ in opt_model.named_parameters()]
                # Find full names that contain any of the partial names
                non_llm_trainable_modules_names = [
                    name for name in all_modules_names if any(partial in name for partial in self.non_llm_trainable_modules)
                ]
                ## non-llm custom projection modules
                input_projection_non_llm_trainable_modules_names = [n for n in non_llm_trainable_modules_names if 'llm_input_adapter' in n]
                output_projection_non_llm_trainable_modules_names = [n for n in non_llm_trainable_modules_names if 'llm_output_adapter' in n]
                projection_modules_names = input_projection_non_llm_trainable_modules_names+output_projection_non_llm_trainable_modules_names
                ## non-llm custom backbone encoder and decoder modules
                other_non_llm_trainable_modules_names = [n for n in non_llm_trainable_modules_names if n not in projection_modules_names]
                
                projection_params = []
                custom_modules_params = []
                llm_decay_params = []
                llm_no_decay_params = []
                for n, p in opt_model.named_parameters():
                    if n in projection_modules_names and p.requires_grad:
                        projection_params.append(p)
                    elif n in other_non_llm_trainable_modules_names and p.requires_grad:
                        custom_modules_params.append(p)
                    elif p.requires_grad: # llm parameters that requires grad
                        if n in decay_parameters:
                            llm_decay_params.append(p)
                        else:   
                            llm_no_decay_params.append(p)
                
                optimizer_grouped_parameters = [
                    {
                        "params": projection_params,
                        "weight_decay": 0.0,
                        # "lr": self.args.learning_rate,
                        "lr": self.args.learning_rate,
                    },
                    {
                        "params": custom_modules_params,
                        "weight_decay": 0.0,
                        "lr": self.args.learning_rate*0.1,
                    },
                    {
                        "params": llm_decay_params,
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.learning_rate,
                    },
                    {
                        "params": llm_no_decay_params,
                        "weight_decay": 0.0,
                        "lr": self.args.learning_rate,
                    },
                ]
            else:                        
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p 
                            for n, p in opt_model.named_parameters() 
                            if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p 
                            for n, p in opt_model.named_parameters() 
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )

            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum(
                            {
                                p.data_ptr(): p.numel() for p in module.parameters()
                            }.values()
                        )
                        manager.register_module_override(
                            module, "weight", {"optim_bits": 32}
                        )
        return self.optimizer
    
    def return_merged_model(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
        ):
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args = self.args

        self.is_in_train = True

        # Attach NEFTune hooks if necessary
        if self.neftune_noise_alpha is not None:
            self.model = self._activate_neftune(self.model)

        # do_train is not a reliable argument, as it might not be set and .train() still called, so
        # the following is a workaround:
        if (args.fp16_full_eval or args.bf16_full_eval) and not args.do_train and not self.is_model_parallel:
            self._move_model_to_device(self.model, args.device)

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(f"train() got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)
        self._train_batch_size = self.args.train_batch_size

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        if resume_from_checkpoint is not None:
            if not is_sagemaker_mp_enabled() and not self.is_deepspeed_enabled and not self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint)
            
            # self.model.merge_and_unload()
            self._move_model_to_device(self.model, args.device)
            return self.model
        else:
            raise "No checkpoint provided"
            # inner_training_loop(
            #         args=args,
            #         resume_from_checkpoint=resume_from_checkpoint,
            #         trial=trial,
            #         ignore_keys_for_eval=ignore_keys_for_eval,
            #     )

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Saving additional modules
        
        if not isinstance(self.model.llm, supported_classes):
            if state_dict is None:
                state_dict = self.model.llm.state_dict()

            if isinstance(self.accelerator.unwrap_model(self.model).llm, supported_classes):
                self.accelerator.unwrap_model(self.model).llm.save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
                print(f"Saved LLM to {output_dir}/")
                if not self.model.use_mtr:
                    components_to_save = {
                        "gameformer_model": self.accelerator.unwrap_model(self.model).gameformer_model,
                        "llm_input_adapter": self.accelerator.unwrap_model(self.model).llm_input_adapter,
                        "llm_output_adapter": self.accelerator.unwrap_model(self.model).llm_output_adapter,
                    }
                else:
                    components_to_save = {
                        "mtr": self.accelerator.unwrap_model(self.model).mtr,
                        "llm_input_adapter": self.accelerator.unwrap_model(self.model).llm_input_adapter,
                        "llm_input_adapter2": self.accelerator.unwrap_model(self.model).llm_input_adapter2,
                        "llm_output_adapter": self.accelerator.unwrap_model(self.model).llm_output_adapter,
                    }
                for name, component in components_to_save.items():
                    if hasattr(component, "state_dict"):
                        # Save the component's state_dict
                        component_path = os.path.join(output_dir, f"{name}.pth")
                        torch.save(component.state_dict(), component_path)
                        print(f"Saved {name} to {component_path}")
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.llm.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors, save_embedding_layers=True
            )
            print(f"Saved LLM to {output_dir}/")
            if self.model.use_mtr:
                components_to_save = {
                    "mtr": self.model.mtr,
                    "llm_input_adapter": self.model.llm_input_adapter,
                    "llm_input_adapter2": self.model.llm_input_adapter2,
                    "llm_output_adapter": self.model.llm_output_adapter,
                }
            else:
                components_to_save = {
                    "gameformer_model": self.model.gameformer_model,
                    "llm_input_adapter": self.model.llm_input_adapter,
                    "llm_output_adapter": self.model.llm_output_adapter,
                }
            for name, component in components_to_save.items():
                if hasattr(component, "state_dict"):
                    # Save the component's state_dict
                    component_path = os.path.join(output_dir, f"{name}.pth")
                    torch.save(component.state_dict(), component_path)
                    print(f"Saved {name} to {component_path}")

        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        
    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        if model is None:
            model = self.model.llm

        config_file = os.path.join(resume_from_checkpoint, CONFIG_NAME)
        adapter_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_WEIGHTS_NAME)
        adapter_safe_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_SAFE_WEIGHTS_NAME)
        weights_file = os.path.join(resume_from_checkpoint, WEIGHTS_NAME)
        weights_index_file = os.path.join(resume_from_checkpoint, WEIGHTS_INDEX_NAME)
        safe_weights_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_NAME)
        safe_weights_index_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_INDEX_NAME)
        is_fsdp_ckpt = os.path.isdir(resume_from_checkpoint) and (
            # this checks the FSDP state dict when `SHARDED_STATE_DICT` is used
            any(
                FSDP_MODEL_NAME in folder_name
                for folder_name in os.listdir(resume_from_checkpoint)
                if os.path.isdir(os.path.join(resume_from_checkpoint, folder_name))
            )
            # this checks the FSDP state dict when `FULL_STATE_DICT` is used
            or os.path.isfile(os.path.join(resume_from_checkpoint, f"{FSDP_MODEL_NAME}.bin"))
        )
        # if multiple adapters exist, they get saved in sub directories
        adapter_subdirs = (
            [
                folder_name
                for folder_name in os.listdir(resume_from_checkpoint)
                if os.path.isdir(os.path.join(resume_from_checkpoint, folder_name))
                and (
                    os.path.isfile(os.path.join(resume_from_checkpoint, folder_name, ADAPTER_WEIGHTS_NAME))
                    or os.path.isfile(os.path.join(resume_from_checkpoint, folder_name, ADAPTER_SAFE_WEIGHTS_NAME))
                )
            ]
            if os.path.isdir(resume_from_checkpoint)
            else []
        )

        if is_fsdp_ckpt and not self.is_fsdp_enabled:
            raise ValueError(f"Checkpoint found at {resume_from_checkpoint} is only supported when using PyTorch FSDP")

        if not (
            any(
                os.path.isfile(f)
                for f in [
                    weights_file,
                    safe_weights_file,
                    weights_index_file,
                    safe_weights_index_file,
                    adapter_weights_file,
                    adapter_safe_weights_file,
                ]
            )
            or is_fsdp_ckpt
            or adapter_subdirs
        ):
            raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

        logger.info(f"Loading model from {resume_from_checkpoint}.")

        if os.path.isfile(config_file):
            config = PretrainedConfig.from_json_file(config_file)
            checkpoint_version = config.transformers_version
            if checkpoint_version is not None and checkpoint_version != __version__:
                logger.warning(
                    f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                    f"Transformers but your current version is {__version__}. This is not recommended and could "
                    "yield to errors or unwanted behaviors."
                )

        if os.path.isfile(weights_file) or os.path.isfile(safe_weights_file) or is_fsdp_ckpt:
            weights_only_kwarg = {"weights_only": True} if is_torch_greater_or_equal_than_1_13 else {}
            # If the model is on the GPU, it still works!
            if is_sagemaker_mp_enabled():
                if os.path.isfile(os.path.join(resume_from_checkpoint, "user_content.pt")):
                    # If the 'user_content.pt' file exists, load with the new smp api.
                    # Checkpoint must have been saved with the new smp api.
                    smp.resume_from_checkpoint(
                        path=resume_from_checkpoint, tag=WEIGHTS_NAME, partial=False, load_optimizer=False
                    )
                else:
                    # If the 'user_content.pt' file does NOT exist, load with the old smp api.
                    # Checkpoint must have been saved with the old smp api.
                    if hasattr(self.args, "fp16") and self.args.fp16 is True:
                        logger.warning(
                            "Enabling FP16 and loading from smp < 1.10 checkpoint together is not suppported."
                        )
                    state_dict = torch.load(
                        weights_file,
                        map_location="cpu",
                        **weights_only_kwarg,
                    )
                    # Required for smp to not auto-translate state_dict from hf to smp (is already smp).
                    state_dict["_smp_is_partial"] = False
                    load_result = model.load_state_dict(state_dict, strict=True)
                    # release memory
                    del state_dict
            elif self.is_fsdp_enabled:
                load_fsdp_model(
                    self.accelerator.state.fsdp_plugin,
                    self.accelerator,
                    model,
                    resume_from_checkpoint,
                    **_get_fsdp_ckpt_kwargs(),
                )
            else:
                # We load the model state dict on the CPU to avoid an OOM error.
                if self.args.save_safetensors and os.path.isfile(safe_weights_file):
                    state_dict = safetensors.torch.load_file(safe_weights_file, device="cpu")
                else:
                    state_dict = torch.load(
                        weights_file,
                        map_location="cpu",
                        **weights_only_kwarg,
                    )

                # workaround for FSDP bug https://github.com/pytorch/pytorch/issues/82963
                # which takes *args instead of **kwargs
                load_result = model.load_state_dict(state_dict, False)
                # release memory
                del state_dict
                self._issue_warnings_after_load(load_result)

        # Load adapters following PR # 24096
        elif _is_peft_model(model):
            # If train a model using PEFT & LoRA, assume that adapter have been saved properly.
            # TODO: in the future support only specific min PEFT versions
            if (hasattr(model, "active_adapter") or hasattr(model, "active_adapters")) and hasattr(
                model, "load_adapter"
            ):
                if os.path.exists(resume_from_checkpoint):
                    # For BC for older PEFT versions
                    if hasattr(model, "active_adapters"):
                        active_adapters = model.active_adapters
                        if len(active_adapters) > 1:
                            logger.warning("Multiple active adapters detected will only consider the first adapter")
                        active_adapter = active_adapters[0]
                    else:
                        active_adapter = model.active_adapter

                    if adapter_subdirs:
                        for subdir_name in adapter_subdirs:
                            peft_id = os.path.join(resume_from_checkpoint, subdir_name)
                            model.load_adapter(peft_id, subdir_name, is_trainable=(subdir_name == active_adapter))
                        model.set_adapter(active_adapter)
                    else:
                        model.load_adapter(resume_from_checkpoint, active_adapter, is_trainable=True)
                else:
                    logger.warning(
                        "The intermediate checkpoints of PEFT may not be saved correctly, "
                        f"consider using a custom callback to save {ADAPTER_WEIGHTS_NAME} in corresponding saving folders. "
                        "Check some examples here: https://github.com/huggingface/peft/issues/96"
                    )
            else:
                logger.warning("Could not load adapter model, make sure to have `peft>=0.3.0` installed")
        else:
            # We load the sharded checkpoint
            load_result = load_sharded_checkpoint(
                model, resume_from_checkpoint, strict=is_sagemaker_mp_enabled(), prefer_safe=self.args.save_safetensors
            )
            if not is_sagemaker_mp_enabled():
                self._issue_warnings_after_load(load_result)

        ## Loading other non llm modules
        print('Loading other moduels:')
        self.model.llm = model
        if self.model.use_mtr:
            components_to_load = {
                    "mtr": self.model.mtr,
                    "llm_input_adapter": self.model.llm_input_adapter,
                    "llm_input_adapter2": self.model.llm_input_adapter2,
                    "llm_output_adapter": self.model.llm_output_adapter,
                }
        else:
            components_to_load = {
                    "gameformer_model": self.model.gameformer_model,
                    "llm_input_adapter": self.model.llm_input_adapter,
                    "llm_output_adapter": self.model.llm_output_adapter,
                }
        for name, component in components_to_load.items():
            if hasattr(component, "state_dict"):
                weights_file = os.path.join(resume_from_checkpoint, name+'.pth')
                state_dict_ = torch.load(
                        weights_file,
                        map_location="cpu",
                        weights_only= True,
                    )
                # Load the component's state_dict
                print(f"Loading {name}: {weights_file}")
                msg = component.load_state_dict(state_dict_, True)
                print(msg)
        print('Loading checkpoint done')
       
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        # import pdb; pdb.set_trace()
        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = (
                self.processing_class.model_input_names[0] if self.processing_class is not None else None
            )
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )
        # NEW CUSTOM CODE: Check if the dataset has sample weights for weighted sampling
        elif hasattr(self.train_dataset, 'sample_weights') and self.train_dataset.sample_weights is not None:
            return torch.utils.data.WeightedRandomSampler(
            weights=self.train_dataset.sample_weights,
            num_samples=len(self.train_dataset),
            replacement=True
            )
        # in case distributed sampling didn't work (all loader samples the same data), use the following
        # is_distributed = self.args.local_rank != -1
        # if is_distributed:
        #     # For distributed training with weights
        #     from torch.utils.data.distributed import DistributedSampler
            
        #     # Create a custom weighted distributed sampler
        #     return WeightedDistributedSampler(
        #         self.train_dataset, 
        #         weights=self.train_dataset.sample_weights,
        #         num_replicas=self.args.world_size,
        #         rank=self.args.local_rank
        #     )
        # End of new custom code
        else:
            return RandomSampler(self.train_dataset)

