import argparse
import pretty_midi
import musical_project
import os

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file_path", type=str)
    parser.add_argument("output_directory_path", type=str)

    args = parser.parse_args()

    input_file_path = args.input_file_path
    output_directory_path = args.output_directory_path

    pretty_midi_data = pretty_midi.PrettyMIDI(input_file_path)
    musical_project_data = musical_project.encode_midi(pretty_midi_data)

    input_file_basename = os.path.basename(input_file_path)
    input_file_name_without_extention, _ = os.path.splitext(input_file_basename)

    if not os.path.exists(output_directory_path):
        os.mkdir(output_directory_path)

    piece_directory_path = os.path.join(output_directory_path, input_file_name_without_extention)

    if not os.path.exists(piece_directory_path):
        os.mkdir(piece_directory_path)


    for index, track in enumerate(tqdm(musical_project_data, desc="tracks", leave=False)):
        pretty_midi_track_data = musical_project.decode_midi(track)

        track_file_name = f"Track {index:02}.mid"
        pretty_midi_track_data.write(os.path.join(piece_directory_path, track_file_name))


if __name__ == "__main__":
    main()
