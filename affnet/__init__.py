# --------------------------------------------------------
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

import argparse


from .anchor_generator import arguments_anchor_gen
from .image_projection_layers import arguments_image_projection_head
from .layers import arguments_nn_layers
from .matcher_det import arguments_box_matcher
from .misc.averaging_utils import arguments_ema, EMA
from .models import arguments_model, get_model
from .neural_augmentor import arguments_neural_augmentor

def extend_selected_args_with_prefix(
    parser: argparse.ArgumentParser, check_string: str, add_prefix: str
) -> argparse.ArgumentParser:
    """
    Helper function to add a prefix to certain arguments.
    An example use case is distillation, where we want to add --teacher as a prefix to all --model.* arguments
    """
    # all arguments are stored as actions
    options = parser._actions

    for option in options:
        option_strings = option.option_strings
        # option strings are stored as a list
        for option_string in option_strings:
            if option_string.split(".")[0] == check_string:
                parser.add_argument(
                    add_prefix + option.dest.replace("_", "-"),
                    nargs="?"
                    if isinstance(option, argparse._StoreTrueAction)
                    else option.nargs,
                    const=option.const,
                    default=option.default,
                    type=option.type,
                    choices=option.choices,
                    help=option.help,
                    metavar=option.metavar,
                )
    return parser

def modeling_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # model arguments
    parser = arguments_model(parser)
    # neural network layer argumetns
    parser = arguments_nn_layers(parser)
    # EMA arguments
    parser = arguments_ema(parser)
    # anchor generator arguments (for object detection)
    parser = arguments_anchor_gen(parser)
    # box matcher arguments (for object detection)
    parser = arguments_box_matcher(parser)
    # image projection head arguments (usually for multi-modal tasks)
    parser = arguments_image_projection_head(parser)
    # neural aug arguments
    parser = arguments_neural_augmentor(parser)

    # Add teacher as a prefix to enable distillation tasks
    # keep it as the last entry
    parser = extend_selected_args_with_prefix(
        parser, check_string="--model", add_prefix="--teacher."
    )

    return parser
