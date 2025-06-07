from .encoder import *
from .decoder import *

NUM_PITCHES = 128

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, size):
        super().__init__()

        self.proj = nn.Linear(d_model, size)

        for layer in [self.proj]:
            nn.init.xavier_uniform_(layer.weight)

            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)

class Model(nn.Module):
    def __init__(self, num_features=4, d_model=128, num_layers=6, d_ff=512, dropout=0.1):
        super().__init__()

        self.num_features = num_features

        self.encoder_s = Encoder(num_features, d_model, dropout)
        self.encoder_e = Encoder(num_features, d_model, dropout)
        self.encoder_p = Encoder(num_features, d_model, dropout)
        self.encoder_v = Encoder(num_features, d_model, dropout)

        self.decoder_s = Decoder(2, d_model, d_ff, num_layers, dropout)
        self.decoder_e = Decoder(3, d_model, d_ff, num_layers, dropout)
        self.decoder_p = Decoder(1, d_model, d_ff, num_layers, dropout)
        self.decoder_v = Decoder(4, d_model, d_ff, num_layers, dropout)

        self.proj_s = nn.Linear(d_model, 1)
        self.proj_e = nn.Linear(d_model, 1)
        self.proj_p = ProjectionLayer(d_model, NUM_PITCHES)
        self.proj_v = nn.Linear(d_model, 1)

        for layer in [self.proj_s, self.proj_e, self.proj_v]:
            nn.init.xavier_uniform_(layer.weight)

            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        self.float()

    # modes: "pitch", "start", "end", "velocity"
    def forward(self, ipt, mode="pitch", pitches=None, starts=None, ends=None, gen_length=800):
        x = None

        match mode:
            case "pitch":
                x = self.encoder_p(ipt)
            case "start":
                x = self.encoder_s(ipt)
            case "end":
                x = self.encoder_e(ipt)
            case "velocity":
                x = self.encoder_v(ipt)

        assert x is not None

        x = x[:, -1, :].unsqueeze(dim=1).expand(-1, gen_length, -1)

        positions = torch.arange(0, gen_length, 1).unsqueeze(dim=0).unsqueeze(dim=-1).expand(x.size(0), -1, -1).float().to(ipt.device)

        match mode:
            case "pitch":
                context = positions
                context = context.float()

                x = self.decoder_p(x, context)

                return self.proj_p(x)

            case "start":
                assert pitches is not None

                pitches = pitches.float().to(ipt.device)

                #print(positions.shape)
                #print(pitches.shape)

                context = torch.dstack([
                    positions,
                    pitches
                ])
                context = context.float()

                x = self.decoder_s(x, context)

                return self.proj_s(x)

            case "end":
                assert pitches is not None
                assert starts is not None

                pitches = pitches.float().to(ipt.device)
                starts = starts.float().to(ipt.device)

                context = torch.dstack([
                    positions,
                    pitches,
                    starts
                ])
                context = context.float()

                x = self.decoder_e(x, context)

                return self.proj_e(x)

            case "velocity":
                assert pitches is not None
                assert starts is not None
                assert ends is not None

                pitches = pitches.float().to(ipt.device)
                starts = starts.float().to(ipt.device)
                ends = ends.float().to(ipt.device)

                context = torch.dstack([
                    positions,
                    pitches,
                    starts,
                    ends
                ])
                context = context.float()

                x = self.decoder_v(x, context)

                return self.proj_v(x)

        raise RuntimeError("why!?")
