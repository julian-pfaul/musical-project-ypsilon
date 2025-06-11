import torch
import argparse
import json
import os
import pretty_midi
import numpy
import shutil

import musical_project

from tqdm import tqdm

print = lambda x: tqdm.write(f"{x}")


def load_run_data(path):
    with open(path, 'r') as f:
        data = json.load(f)

    return data


def load_model(run_data, path):
    model = musical_project.ypsilon.Model(
        run_data["model"]["num_features"],
        run_data["model"]["d_model"],
        run_data["model"]["num_layers"],
        run_data["model"]["d_ff"],
        run_data["model"]["dropout"]
    )

    if not os.path.exists(path):
        raise RuntimeError(f"{path} doesn't exist")

    state_dict = torch.load(path, weights_only=True)

    model.load_state_dict(state_dict)

    return model


def generate(run_data, model, input_file_path, output_file_path, device):
    piece = musical_project.encode_midi(pretty_midi.PrettyMIDI(input_file_path))

    context_length = run_data["model"]["context_length"]
    maximum_generation_length = run_data["model"]["maximum_generation_length"]

    piece = piece.squeeze().to(device)
    inputs = piece[:context_length].unsqueeze(dim=0)

    with torch.no_grad():
        pitch_dists = model(inputs, mode="pitch", pitches=None, starts=None, ends=None, gen_length=maximum_generation_length)
        _, m = torch.max(pitch_dists, dim=-1)
        pitches = m.unsqueeze(dim=-1)

        starts = model(inputs, mode="start", pitches=pitches, starts=None, ends=None, gen_length=maximum_generation_length)

        ends = model(inputs, mode="end", pitches=pitches, starts=starts, ends=None, gen_length=maximum_generation_length)

        velocities = model(inputs, mode="velocity", pitches=pitches, starts=starts, ends=ends, gen_length=maximum_generation_length)

        inputs = inputs.squeeze()
        outputs = torch.dstack([starts, ends, pitches, velocities]).squeeze()

        piece = torch.vstack([inputs, outputs])

        pretty_midi_data = musical_project.decode_midi(piece)
        pretty_midi_data.write(output_file_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_data_path", type=str)
    parser.add_argument("model_file_path", type=str)
    parser.add_argument("input_file_path", type=str)
    parser.add_argument("output_file_path", type=str)

    args = parser.parse_args()

    run_data_path = args.run_data_path
    model_file_path = args.model_file_path
    input_file_path = args.input_file_path
    output_file_path = args.output_file_path

    run_data = load_run_data(run_data_path)

    device = torch.device("cuda")

    model = load_model(run_data, model_file_path)
    model = model.to(device)

    generate(run_data, model, input_file_path, output_file_path, device)

if __name__ == "__main__":
    main()










