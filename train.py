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


def plot(x, y, rows=None, columns=None):
    """
    x, y list of values on x- and y-axis
    plot those values within canvas size (rows and columns)
    """
    def_row, def_col = get_terminal_size()
    rows = rows if rows else def_row
    columns = columns if columns else def_col

    rows, columns = columns, rows
    # offset for caption
    rows -= 4

    # Scale points such that they fit on canvas
    x_scaled = scale(x, columns)
    y_scaled = scale(y, rows)

    points = list(zip(x_scaled, y_scaled))

    x_positions = []
    y_positions = []

    for p1, p2 in zip(points, points[1:]):
        p1x = p1[0]
        p1y = p1[1]
        p2x = p2[0]
        p2y = p2[1]

        delta_x = p2x - p1x
        delta_y = p2y - p1y

        y_step = delta_y / delta_x

        for step in range(delta_x):
            x_pos = p1x + step
            y_pos = p1y + step*y_step

            x_positions.append(int(x_pos))
            y_positions.append(int(y_pos))


    # Create empty canvas
    canvas = [[" " for _ in range(columns)] for _ in range(rows)]

    # Add scaled points to canvas
    for ix, iy in zip(x_positions, y_positions):
        canvas[rows - iy - 1][ix] = "*"

    canvas = "\n".join(["".join(row) for row in canvas])

    # Print scale
    print(
        "".join(
            [
                "\n",
                canvas,
                "\nMin x: ",
                str(min(x)),
                " Max x: ",
                str(max(x)),
                " Min y: ",
                str(min(y)),
                " Max y: ",
                str(max(y)),
            ]
        )
    )


def scale(x, length):
    """
    Scale points in 'x', such that distance between
    max(x) and min(x) equals to 'length'. min(x)
    will be moved to 0.
    """
    s = (
        float(length - 1) / (max(x) - min(x))
        if x and max(x) - min(x) != 0
        else length
    )

    return [int((i - min(x)) * s) for i in x]


# FROM terminalplot
def get_terminal_size():
    return shutil.get_terminal_size()


def load_run_data(path):
    with open(path, 'r') as f:
        data = json.load(f)

    return data


def load_model(run_data, directory):
    model = musical_project.ypsilon.Model(
        run_data["model"]["num_features"],
        run_data["model"]["d_model"],
        run_data["model"]["num_layers"],
        run_data["model"]["d_ff"],
        run_data["model"]["dropout"]
    )

    if not os.path.exists(directory):
        os.mkdir(directory)

        save_model(directory, model)

        return model

    state_dict_path = os.path.join(directory, "model_state_dict")
    state_dict = torch.load(state_dict_path, weights_only=True)

    model.load_state_dict(state_dict)

    return model


def save_model(directory, model):
    model.eval()

    state_dict_path = os.path.join(directory, "model_state_dict")
    torch.save(model.state_dict(), state_dict_path)


def load_checkpoint(directory):
    next_epoch_path = os.path.join(directory, "next_epoch")
    model_state_dict_path = os.path.join(directory, "model_state_dict")
    optimizer_state_dict = os.path.join(directory, "optimizer_state_dict")

    next_epoch = torch.load(next_epoch_path, weights_only=True)
    model_state_dict = torch.load(model_state_dict_path, weights_only=True)
    optimizer_state_dict = torch.load(optimizer_state_dict, weights_only=True)

    checkpoint = dict()
    checkpoint["next_epoch"] = next_epoch
    checkpoint["model_state_dict"] = model_state_dict
    checkpoint["optimizer_state_dict"] = optimizer_state_dict

    return checkpoint


def save_checkpoint(directory, next_epoch, model, optimizer, average_loss=-1.0):
    if not os.path.exists(directory):
        os.mkdir(directory)

    next_epoch_path = os.path.join(directory, "next_epoch")
    model_state_dict_path = os.path.join(directory, "model_state_dict")
    optimizer_state_dict_path = os.path.join(directory, "optimizer_state_dict")
    average_loss_path = os.path.join(directory, "average_loss")

    torch.save(next_epoch, next_epoch_path)
    torch.save(model.state_dict(), model_state_dict_path)
    torch.save(optimizer.state_dict(), optimizer_state_dict_path)
    torch.save(average_loss, average_loss_path)


def apply_checkpoint(model, optimizer, checkpoint):
    next_epoch = checkpoint["next_epoch"]
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return next_epoch, model, optimizer


def load_raw_dataset(run_data, raw_dataset_path):
    if os.path.exists(raw_dataset_path):
        raw_dataset = torch.load(raw_dataset_path, weights_only=False)

        return raw_dataset

    data_directory = run_data["directories"]["data"]

    data_file_names = os.listdir(data_directory)
    data_file_paths = [os.path.join(data_directory, data_file_name) for data_file_name in data_file_names]
    data_file_paths = [data_file_path for data_file_path in data_file_paths if run_data["dataset"]["filter"] in os.path.basename(data_file_path)]

    pieces = []

    for path in tqdm(data_file_paths, desc="midi files", leave=False):
        try:
            midi = pretty_midi.PrettyMIDI(path)

            base_name = os.path.basename(path)
            name, _ = os.path.splitext(base_name)

            piece = musical_project.encode_midi(midi)

            pieces.append(piece)

        except Exception as e:
            print(e)

    raw_dataset = musical_project.ypsilon.RawDataset(pieces)

    torch.save(raw_dataset, raw_dataset_path)

    return raw_dataset


def epoch_generator(initial_epoch, maximum_epoch):
    epoch = initial_epoch

    while initial_epoch <= maximum_epoch:
        yield epoch

        epoch += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_data_path", type=str)

    args = parser.parse_args()

    run_data_path = args.run_data_path

    run_data = load_run_data(run_data_path)

    for directory_name, directory_path in tqdm(run_data["directories"].items(), desc="directories",leave=False):
        if not os.path.exists(directory_path):
            os.mkdir(directory_path)

    model_directory = os.path.join(run_data["directories"]["model"], run_data["model"]["name"])

    device = torch.device("cuda")

    next_epoch = 0
    model = load_model(run_data, model_directory)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=run_data["optimizer"]["lr"])

    checkpoints_directory = os.path.join(model_directory, "checkpoint")

    if not os.path.exists(checkpoints_directory):
        os.mkdir(checkpoints_directory)

    checkpoint_directory = os.path.join(checkpoints_directory, "init")

    if not os.path.exists(checkpoint_directory):
        save_checkpoint(checkpoint_directory, next_epoch, model, optimizer)

    checkpoint_directory = os.path.join(checkpoints_directory, "main")

    if os.path.exists(checkpoint_directory):
        checkpoint = load_checkpoint(checkpoint_directory)

        next_epoch, model, optimizer = apply_checkpoint(model, optimizer, checkpoint)
    else:
        save_checkpoint(checkpoint_directory, next_epoch, model, optimizer)


    raw_dataset_path = os.path.join(run_data["directories"]["dataset"], run_data["dataset"]["name"])
    raw_dataset = load_raw_dataset(run_data, raw_dataset_path)

    dataset = musical_project.ypsilon.TrackDataset(raw_dataset, run_data["model"]["context_length"])

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=run_data["batch_size"], shuffle=True)

    initial_epoch = next_epoch
    maximum_epoch = run_data["maximum_epoch"]

    maximum_generation_length = run_data["model"]["maximum_generation_length"]

    l1_loss_fn = torch.nn.L1Loss(reduction="mean")
    cross_entropy_loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")

    epoch_iterator = tqdm(epoch_generator(initial_epoch, maximum_epoch), desc="epoches", leave=False, initial=initial_epoch)

    average_losses = []

    for epoch in epoch_iterator:
        model.train()

        batch_iterator = tqdm(data_loader, desc="batches", leave=False)

        losses = []

        for batch in batch_iterator:
            inputs, s_labels, e_labels, p_labels, v_labels = batch

            inputs = inputs.cuda()
            s_labels = s_labels[:, :maximum_generation_length, :].cuda()
            e_labels = e_labels[:, :maximum_generation_length, :].cuda()
            p_labels = p_labels[:, :maximum_generation_length, :].cuda()
            v_labels = v_labels[:, :maximum_generation_length, :].cuda()

            p_outputs = model(inputs, "pitch",    pitches=None,     starts=None,     ends=None,     gen_length=maximum_generation_length)
            s_outputs = model(inputs, "start",    pitches=p_labels, starts=None,     ends=None,     gen_length=maximum_generation_length)
            e_outputs = model(inputs, "end",      pitches=p_labels, starts=s_labels, ends=None,     gen_length=maximum_generation_length)
            v_outputs = model(inputs, "velocity", pitches=p_labels, starts=s_labels, ends=e_labels, gen_length=maximum_generation_length)

            s_loss = l1_loss_fn(s_outputs, s_labels)
            e_loss = l1_loss_fn(e_outputs, e_labels)
            p_loss = cross_entropy_loss_fn(p_outputs.float().view(-1, 128), p_labels.long().view(-1))
            v_loss = l1_loss_fn(v_outputs, v_labels)

            loss = s_loss + e_loss + p_loss + v_loss

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            losses.append(loss.item())

        average_loss = numpy.average(numpy.array(losses))
        average_losses.append(average_loss)

        epoch_iterator.set_postfix({f"average loss": f"{average_loss:6.3f}"})

        num_average_losses = len(average_losses)
        maximum_displayed_average_losses = run_data["maximum_displayed_average_losses"]

        print("\033[H\033[J")
        plot(range(epoch-min(num_average_losses - 1, maximum_displayed_average_losses - 1), epoch+1), average_losses[-maximum_displayed_average_losses:])

        save_model(model_directory, model)

        if epoch % run_data["checkpoint_interval"] == 0:
            checkpoint_directory = os.path.join(checkpoints_directory, f"epoch-{epoch}")
            save_checkpoint(checkpoint_directory, epoch+1, model, optimizer, average_loss)

        checkpoint_directory = os.path.join(checkpoints_directory, "main")
        save_checkpoint(checkpoint_directory, epoch+1, model, optimizer, average_loss)

if __name__ == "__main__":
    main()










