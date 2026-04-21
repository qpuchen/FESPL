import os
from argparse import ArgumentParser

import torch


def get_swa_model_(model_dir, starting_model_id="0", ending_model_id="11"):
    save_dir = model_dir
    model_dir = model_dir
    starting_id = int(starting_model_id)
    ending_id = int(ending_model_id)
    model_names = list(range(starting_id, ending_id + 1))
    model_dirs = [os.path.join(model_dir, "epoch_" + str(i) + ".pth") for i in model_names]
    models = [torch.load(model_dir) for model_dir in model_dirs]
    model_num = len(models)
    # model_keys = models[-1]['state_dict'].keys()
    # state_dict = models[-1]['state_dict']
    model_keys = models[-1].keys()
    state_dict = models[-1]
    new_state_dict = state_dict.copy()
    ref_model = models[-1]

    for key in model_keys:
        sum_weight = 0.0
        for m in models:
            # sum_weight += m['state_dict'][key]
            sum_weight += m[key]
        avg_weight = sum_weight / model_num
        new_state_dict[key] = avg_weight
    ref_model = new_state_dict
    # ref_model['state_dict'] = new_state_dict
    save_model_name = "swa_" + starting_model_id + "-" + ending_model_id + ".pth"
    if save_dir is not None:
        save_dir = os.path.join(save_dir, save_model_name)
    else:
        save_dir = os.path.join(model_dir, save_model_name)
    torch.save(ref_model, save_dir)
    print("Model is saved at", save_dir)
    return ref_model


def main():
    parser = ArgumentParser()
    parser.add_argument("model_dir", help="the directory where checkpoints are saved")
    parser.add_argument(
        "starting_model_id",
        type=int,
        help="the id of the starting checkpoint for averaging, e.g. 1",
    )
    parser.add_argument(
        "ending_model_id",
        type=int,
        help="the id of the ending checkpoint for averaging, e.g. 12",
    )
    parser.add_argument("--save_dir", default=None, help="the directory for saving the SWA model")
    args = parser.parse_args()

    model_dir = args.model_dir
    starting_id = int(args.starting_model_id)
    ending_id = int(args.ending_model_id)
    model_names = list(range(starting_id, ending_id + 1))
    model_dirs = [os.path.join(model_dir, "epoch_" + str(i) + ".pth") for i in model_names]
    models = [torch.load(model_dir) for model_dir in model_dirs]
    model_num = len(models)
    model_keys = models[-1]["state_dict"].keys()
    state_dict = models[-1]["state_dict"]
    new_state_dict = state_dict.copy()
    ref_model = models[-1]

    for key in model_keys:
        sum_weight = 0.0
        for m in models:
            sum_weight += m["state_dict"][key]
        avg_weight = sum_weight / model_num
        new_state_dict[key] = avg_weight
    ref_model["state_dict"] = new_state_dict
    save_model_name = "swa_" + args.starting_model_id + "-" + args.ending_model_id + ".pth"
    if args.save_dir is not None:
        save_dir = os.path.join(args.save_dir, save_model_name)
    else:
        save_dir = os.path.join(model_dir, save_model_name)
    torch.save(ref_model, save_dir)
    print("Model is saved at", save_dir)


if __name__ == "__main__":
    main()
