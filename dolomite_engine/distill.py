import logging
import time
from contextlib import AbstractContextManager, nullcontext
from functools import partial

import torch
from torch.distributed import ReduceOp
from torch.distributed.tensor.parallel import loss_parallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .arguments import TrainingArgs
from .checkpointing import save_checkpoint
from .communication import Communication
from .data import ResumableDataLoader, get_next_batch
from .enums import DistributedBackend, FP8Backend, Mode
from .model_wrapper import ModelWrapperForFinetuning, ModelWrapperForPretraining
from .pretrain import main
from .train_utils import get_model_tflops, get_torch_profiler, track_train_metrics, train_step
from .utils import ExperimentsTracker, ProcessGroupManager, is_transformer_engine_available, log_rank_0


if is_transformer_engine_available():
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling, Format


def track_val_metrics(
    global_step: int,
    val_loss: float,
    val_lm_loss: float,
    val_kl_divergence: float,
    experiments_tracker: ExperimentsTracker,
    group_name: str | None = None,
) -> None:
    """tracks metrics like validation loss

    Args:
        global_step (int): global step during training
        val_loss (float): validation loss for the validation data
        experiments_tracker (ExperimentsTracker): metrics tracker
        group_name (str | None): group name for the validation / test set
    """

    message = (
        f"step = {global_step}, val_loss = {val_loss:.4f}, val_lm_loss = {val_lm_loss:.4f}, "
        f"val_KL = {val_kl_divergence:.4f}"
    )
    if group_name is not None:
        message += f", group_name = {group_name}"

    log_rank_0(logging.INFO, message)
    experiments_tracker.track(
        {
            "loss" if group_name is None else f"loss-{group_name}": val_loss,
            "lm_loss" if group_name is None else f"lm_loss-{group_name}": val_lm_loss,
            "KL" if group_name is None else f"KL-{group_name}": val_kl_divergence,
        },
        step=global_step,
        context="val",
    )


def train_step(
    model: ModelWrapperForFinetuning,
    optimizer: Optimizer,
    lr_scheduler: LambdaLR,
    distributed_backend: DistributedBackend,
    train_dataloader: ResumableDataLoader,
    gradient_accumulation_steps: int,
    gradient_clipping: float,
    forward_context: AbstractContextManager,
    backward_context: AbstractContextManager,
) -> tuple[float, float]:
    """runs backpropagation and applies the gradient if at the edge of gradient accumulation boundary

    Args:
        model (ModelWrapperForFinetuning): model
        optimizer (Optimizer): optimizer
        lr_scheduler (LamdaLR): learning rate scheduler
        distributed_backend (DistributedBackend): distributed backend
        train_dataloader (ResumableDataLoader): training dataloader
        gradient_accumulation_steps (int): gradient accumulation steps
        gradient_clipping (float): gradient clipping value
        forward_context (AbstractContextManager): a context that is used for every model forward call
        backward_context (AbstractContextManager): a context that is used for every model backward call

    Returns:
        tuple[float, float]: loss at the current step, grad norm at the current step
    """

    no_sync = nullcontext
    if distributed_backend == DistributedBackend.torch:
        fsdp_algorithm = 2 if hasattr(model, "set_requires_gradient_sync") else 1

        if fsdp_algorithm == 1:
            no_sync = model.no_sync
        else:
            model.set_requires_gradient_sync(False)

    loss = 0
    lm_loss = 0
    kl_divergence = 0
    grad_norm = None
    if distributed_backend == DistributedBackend.torch:
        optimizer.zero_grad()

    with no_sync():
        for _ in range(gradient_accumulation_steps - 1):
            batch = get_next_batch(train_dataloader)
            with forward_context():
                loss_micro_step, lm_loss_step, kl_divergence_step = model(batch)
            loss += loss_micro_step
            lm_loss += lm_loss_step
            kl_divergence += kl_divergence_step

            # compute gradients
            if distributed_backend == DistributedBackend.deepspeed:
                with backward_context():
                    model.backward(loss_micro_step)
                model.step()
            elif distributed_backend == DistributedBackend.torch:
                with backward_context():
                    loss_micro_step.backward()
            else:
                raise ValueError(f"unexpected distributed backend ({distributed_backend})")

    if distributed_backend == DistributedBackend.torch and fsdp_algorithm == 2:
        model.set_requires_gradient_sync(True)

    batch = get_next_batch(train_dataloader)
    with forward_context():
        loss_micro_step, lm_loss_step, kl_divergence_step = model(batch)
    loss += loss_micro_step
    lm_loss += lm_loss_step
    kl_divergence += kl_divergence_step

    # compute gradients
    if distributed_backend == DistributedBackend.deepspeed:
        with backward_context():
            model.backward(loss_micro_step)

        if gradient_clipping is not None:
            grad_norm = model.get_global_grad_norm()

        model.step()
    elif distributed_backend == DistributedBackend.torch:
        with backward_context():
            loss_micro_step.backward()

        if gradient_clipping is not None:
            if fsdp_algorithm == 1:
                grad_norm = model.clip_grad_norm_(gradient_clipping)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

        optimizer.step()
        lr_scheduler.step()
    else:
        raise ValueError(f"unexpected distributed backend ({distributed_backend})")

    loss /= gradient_accumulation_steps
    lm_loss /= gradient_accumulation_steps
    kl_divergence /= gradient_accumulation_steps

    if ProcessGroupManager.get_tensor_parallel_world_size() > 1:
        loss = loss.to_local()
        raise NotImplementedError()

    tensor = torch.stack([loss, lm_loss, kl_divergence])
    torch.distributed.all_reduce(tensor, op=ReduceOp.AVG, group=ProcessGroupManager.get_data_parallel_group())

    tensor = tensor.tolist()
    loss, lm_loss, kl_divergence = tensor
    grad_norm = 0 if grad_norm is None else grad_norm.item()

    return loss, lm_loss, kl_divergence, grad_norm


def train(
    args: TrainingArgs,
    model: ModelWrapperForPretraining,
    optimizer: Optimizer,
    lr_scheduler: LambdaLR,
    train_dataloader: DataLoader,
    val_dataloaders: list[DataLoader],
    test_dataloaders: list[DataLoader],
    experiments_tracker: ExperimentsTracker,
    starting_iteration: int = 0,
) -> None:
    """main training loop for the program

    Args:
        args (TrainingArgs): training args
        model (ModelWrapperForPretraining): model
        optimizer (Optimizer): optimizer
        lr_scheduler (LRScheduler): learning rate scheduler
        train_dataloader (DataLoader): training dataloader
        val_dataloaders (list[DataLoader]): validation dataloaders
        test_dataloaders (list[DataLoader]): test dataloaders
        experiments_tracker (ExperimentsTracker): metrics tracker
        starting_iteration (int): starting iteration
    """

    assert args.distributed_args.fsdp_algorithm == 2, "Distillation is only supported with FSDP-2"

    num_training_steps = args.training_parameters.num_training_steps
    gradient_accumulation_steps = args.training_parameters.gradient_accumulation_steps
    gradient_clipping = args.training_parameters.gradient_clipping

    eval_during_training = args.training_parameters.eval_during_training
    eval_interval = args.training_parameters.eval_interval
    distributed_backend = args.distributed_args.distributed_backend
    save_interval = args.save_args.save_interval
    log_interval = args.logging_args.log_interval

    val_weighted_split_paths = args.datasets[0].class_args.get("val_weighted_split_paths")
    group_names = [None]
    if val_weighted_split_paths is not None:
        group_names = [key for key in val_weighted_split_paths.keys()[0]]

    model.train()

    if eval_during_training:
        eval_steps = args.datasets[0].class_args.get("eval_steps")
        evaluate(val_dataloaders, model, starting_iteration, experiments_tracker, eval_steps, group_names)

    micro_batch_size = args.training_parameters.micro_batch_size
    sequence_length = args.datasets[0].class_args.get("sequence_length")
    global_batch_size = (
        micro_batch_size * gradient_accumulation_steps * ProcessGroupManager.get_data_parallel_world_size()
    )
    tokens_per_batch = global_batch_size * sequence_length

    dp_world_size = ProcessGroupManager.get_data_parallel_world_size()

    # model flops per GPU
    model_flops = (
        get_model_tflops(
            model_class=args.model_args.model_class,
            config=model.config,
            batch_size=global_batch_size,
            sequence_length=sequence_length,
            gradient_checkpointing_method=args.distributed_args.gradient_checkpointing_method,
            gradient_checkpointing_args=args.distributed_args.gradient_checkpointing_args,
        )
        / dp_world_size
    )

    forward_context = (
        partial(
            te.fp8_autocast,
            enabled=True,
            fp8_recipe=DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max"),
        )
        if args.mixed_precision_args.dtype == "fp8" and args.mixed_precision_args.fp8_backend == FP8Backend.nvte
        else nullcontext
    )

    backward_context = loss_parallel if args.distributed_args.tensor_parallel_word_embeddings else nullcontext

    torch_profiler = get_torch_profiler(args.logging_args.torch_profiler_trace_path)

    if torch_profiler is not None:
        torch_profiler.__enter__()

    start_time = time.perf_counter()
    steps_since_start_time = 0
    loss_running_sum = 0
    lm_loss_running_sum = 0
    kl_divergence_running_sum = 0

    global_step = starting_iteration
    while global_step < num_training_steps:
        global_step += 1
        steps_since_start_time += 1

        loss_step, lm_loss_step, kl_divergence_step, grad_norm_step = train_step(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            distributed_backend=distributed_backend,
            train_dataloader=train_dataloader,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_clipping=gradient_clipping,
            forward_context=forward_context,
            backward_context=backward_context,
        )

        loss_running_sum += loss_step
        lm_loss_running_sum += lm_loss_step
        kl_divergence_running_sum += kl_divergence_step

        if torch_profiler is not None:
            torch_profiler.step()

        if global_step % log_interval == 0:
            time_elapsed = time.perf_counter() - start_time
            step_time = time_elapsed / steps_since_start_time

            track_train_metrics(
                global_step=global_step,
                train_loss_step=loss_step,
                grad_norm_step=grad_norm_step,
                current_lr=(
                    model.lr_scheduler.get_lr()[0]
                    if distributed_backend == DistributedBackend.deepspeed
                    else lr_scheduler.get_lr()[0]
                ),
                experiments_tracker=experiments_tracker,
                loss_running_mean=loss_running_sum / log_interval,
                flops=None if model_flops is None else model_flops * steps_since_start_time / time_elapsed,
                billion_tokens_per_day=tokens_per_batch * 86400 / step_time / 1e9,
                step_time=step_time,
                extras={
                    "train_lm_loss (running_mean)": lm_loss_running_sum / log_interval,
                    "train_KL (running_mean)": kl_divergence_running_sum / log_interval,
                },
            )
            start_time = time.perf_counter()
            steps_since_start_time = 0
            loss_running_sum = 0
            lm_loss_running_sum = 0
            kl_divergence_running_sum = 0

        if eval_during_training and (global_step % eval_interval == 0 or global_step == num_training_steps):
            evaluate(val_dataloaders, model, global_step, experiments_tracker, eval_steps, group_names)

        if global_step % save_interval == 0 or global_step == num_training_steps:
            save_checkpoint(
                args,
                model,
                optimizer,
                lr_scheduler,
                None,
                experiments_tracker,
                global_step,
                {"consumed_samples": global_step * micro_batch_size * gradient_accumulation_steps * dp_world_size},
            )

            start_time = time.perf_counter()
            steps_since_start_time = 0

    if eval_during_training:
        evaluate(test_dataloaders, model, global_step, experiments_tracker, eval_steps, group_names)

    if torch_profiler is not None:
        torch_profiler.__exit__()


@torch.no_grad()
def evaluate(
    val_dataloaders: list[DataLoader],
    model: ModelWrapperForPretraining,
    global_step: int,
    experiments_tracker: ExperimentsTracker,
    eval_steps: int,
    group_names: list[str],
) -> float:
    """main validation loop for the program

    Args:
        val_dataloaders (list[DataLoader]): list of validation dataloaders
        model (ModelWrapperForPretraining): model
        global_step (int): global step during training
        experiments_tracker (ExperimentsTracker): metrics tracker
        eval_steps (int): number of steps to run eval for
        group_names (list[str]): names of the datasets in validation/test group

    Returns:
        float: loss at the current step
    """

    tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()

    if tp_world_size > 1:
        # other tensor parallel ranks need to be told if val dataloader is None or not
        is_val_dataloader_none = (
            val_dataloaders is None or len(val_dataloaders) == 0
            if ProcessGroupManager.get_tensor_parallel_rank() == 0
            else None
        )
        is_val_dataloader_none = Communication.broadcast_object(
            is_val_dataloader_none,
            src=ProcessGroupManager.get_tensor_parallel_first_rank(),
            group=ProcessGroupManager.get_tensor_parallel_group(),
        )
    else:
        is_val_dataloader_none = val_dataloaders is None or len(val_dataloaders) == 0

    if is_val_dataloader_none:
        return

    model.eval()

    for group_name, val_dataloader in zip(group_names, val_dataloaders):
        loss_sum = 0
        lm_loss_sum = 0
        kl_divergence_sum = 0

        for _ in range(eval_steps):
            batch = get_next_batch(val_dataloader)
            loss, lm_loss, kl_divergence = model(batch)

            loss_sum += loss
            lm_loss_sum += lm_loss
            kl_divergence_sum += kl_divergence

        loss_mean = loss_sum / eval_steps
        lm_loss_mean = lm_loss_sum / eval_steps
        kl_divergence_mean = kl_divergence_sum / eval_steps

        if tp_world_size > 1:
            loss_mean = loss_mean.to_local()

        tensor = torch.stack([loss_mean, lm_loss_mean, kl_divergence_mean])
        torch.distributed.all_reduce(tensor, op=ReduceOp.AVG, group=ProcessGroupManager.get_data_parallel_group())
        tensor = tensor.tolist()
        loss_mean, lm_loss_mean, kl_divergence_mean = tensor

        track_val_metrics(global_step, loss_mean, lm_loss_mean, kl_divergence_mean, experiments_tracker, group_name)

    model.train()

    return loss_mean


if __name__ == "__main__":
    main(Mode.distillation, train_func=train)