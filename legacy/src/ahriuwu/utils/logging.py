"""Wandb logging utilities for training scripts.

Provides a thin wrapper around wandb with graceful degradation
when wandb is not installed or --no-wandb is passed.
"""

import argparse


def add_wandb_args(parser: argparse.ArgumentParser):
    """Add standard wandb CLI arguments to a parser."""
    group = parser.add_argument_group("wandb")
    group.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable wandb logging (default: True, --no-wandb to disable)",
    )
    group.add_argument(
        "--wandb-project",
        type=str,
        default="ahriuwu",
        help="Wandb project name",
    )
    group.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Wandb entity (team/user)",
    )
    group.add_argument(
        "--wandb-tags",
        type=str,
        nargs="*",
        default=None,
        help="Extra wandb tags",
    )


def init_wandb(args: argparse.Namespace, job_type: str, extra_config: dict | None = None):
    """Initialize wandb run. Returns the run object or None if disabled.

    Args:
        args: Parsed CLI args (must include wandb group args).
        job_type: One of 'tokenizer', 'transformer_tokenizer', 'dynamics', 'agent_finetune'.
        extra_config: Additional config keys to log beyond args.
    """
    if not getattr(args, "wandb", True):
        return None

    try:
        import wandb
    except ImportError:
        print("WARNING: wandb not installed, logging disabled. pip install wandb")
        return None

    config = vars(args).copy()
    if extra_config:
        config.update(extra_config)

    # Build tags
    tags = [job_type]
    if hasattr(args, "model_size"):
        tags.append(args.model_size)
    if hasattr(args, "tokenizer_type"):
        tags.append(f"tok_{args.tokenizer_type}")
    if getattr(args, "wandb_tags", None):
        tags.extend(args.wandb_tags)

    # Build run name
    run_name = getattr(args, "run_name", None)
    if not run_name:
        from datetime import datetime
        ts = datetime.now().strftime("%m%d_%H%M")
        size = getattr(args, "model_size", "")
        run_name = f"{job_type}_{size}_{ts}"

    run = wandb.init(
        project=getattr(args, "wandb_project", "ahriuwu"),
        entity=getattr(args, "wandb_entity", None),
        name=run_name,
        config=config,
        tags=tags,
        job_type=job_type,
        resume="allow",
    )
    return run


def log_step(metrics: dict, step: int):
    """Log per-step metrics to wandb (no-op if wandb not active)."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except ImportError:
        pass


def log_images(images: dict, step: int):
    """Log images to wandb.

    Args:
        images: dict of name -> PIL.Image or torch tensor (CHW, 0-1).
        step: global step.
    """
    try:
        import wandb
        if wandb.run is None:
            return
        log_dict = {}
        for name, img in images.items():
            if hasattr(img, "numpy"):
                # torch tensor CHW -> wandb.Image
                log_dict[name] = wandb.Image(img)
            else:
                log_dict[name] = wandb.Image(img)
        wandb.log(log_dict, step=step)
    except ImportError:
        pass


def finish_wandb():
    """Finish wandb run if active."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass
