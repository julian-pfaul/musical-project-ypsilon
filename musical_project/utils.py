import torch
import pretty_midi


def encode_midi(midi):
    midi_data = []

    for instrument in midi.instruments:
        instrument_data = []

        for note in instrument.notes:
            start = note.start
            end = note.end
            pitch = note.pitch
            velocity = note.velocity

            note_data = torch.tensor([start, end, pitch, velocity])

            instrument_data.append(note_data)

        instrument_data = torch.stack(instrument_data)

        midi_data.append(instrument_data)

    target_length = max([t.size(0) for t in midi_data])
    midi_data = torch.stack([torch.cat([t, torch.zeros(target_length-t.size(0), t.size(1))], dim=0) for t in midi_data])

    return midi_data


def decode_midi(data):
    midi_data = pretty_midi.PrettyMIDI()

    instrument = pretty_midi.Instrument(program=0)

    for note in data:
        start = note[0].item()
        end = note[1].item()
        pitch = int(note[2].item())
        velocity = int(note[3].item())

        if pitch <= 0:
            break

        #print(start, end, end - start)

        midi_note = pretty_midi.Note(velocity=max(0, min(velocity, 127)), pitch=max(0, min(pitch, 127)), start=float(start), end=float(end))

        instrument.notes.append(midi_note)

    midi_data.instruments.append(instrument)

    return midi_data
