import torch

STOP_TOKEN = torch.tensor([0, 0, 0, 0])

class TrackDataset(torch.utils.data.Dataset):
    def __init__(self, raw_dataset, context_size):
        pieces = raw_dataset.pieces

        self.context_size = context_size

        self.tracks = []

        max_length = 0

        for piece in pieces:
            for track in piece:
                max_length = max(max_length, track.size(0))

                self.tracks.append(track)

        for index, track in enumerate(self.tracks):
            track = torch.vstack([track, STOP_TOKEN.repeat(max_length - track.size(0) + 1, 1)])

            self.tracks[index] = track

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        track = self.tracks[idx]

        encoder_input = track[:self.context_size, :].float()
        decoder_label = track[self.context_size:, :].float()

        #print(encoder_input.shape)
        #print(decoder_label.shape)

        s_label = decoder_label[:, 0].unsqueeze(dim=-1)
        e_label = decoder_label[:, 1].unsqueeze(dim=-1)
        p_label = decoder_label[:, 2].unsqueeze(dim=-1)
        v_label = decoder_label[:, 3].unsqueeze(dim=-1)

        return encoder_input, s_label, e_label, p_label, v_label
