import torch
from torch import nn
import torchmetrics
import numpy as np
import torch.nn.functional as F

import benchmark as bench
from benchmark.tasks.MUSDB18.MUSDB18_metrics import GlobalSDR
from openunmix import model, transforms, filtering # For source separation

class MUSDB18Prober(bench.ProberForBertSeqLabel):
    """LSTM Prober for source separation with BERT like 12 layer features (HuBERT, Data2vec).
    This class supports learnable weighted sum over different layers of features.
    """
    def __init__(self, cfg):
        super().__init__(cfg)

        unidirectional = True
        nb_layers = 3

        input_mean = None
        input_scale = None
        self.residual = True
        self.niter = 1
        self.softmask = False
        self.wiener_win_len = 600
        self.test_metrics = []
        self.valid_metrics = []
        self.nb_targets = 1
        self.sample_rate = 24000
        self.nfft = 2048
        self.nhop = 320
        self.nb_output_bins = self.nfft // 2 + 1
        self.max_len = 24000 * 90
        max_bin = None
        if max_bin:
            self.nb_bins = max_bin
        else:
            self.nb_bins = self.nb_output_bins
        
        self.nb_features = 768 # For HuBERT feature
        self.nb_channels = 1 # mono only
        self.nb_bins = 0 

        self.hidden_size = self.hidden_layer_sizes[0]

        assert self.nb_bins == 0
        # if self.nb_bins != 0:
        #     input_mean, input_scale = np.load('/home/music/source-separation/data/stats/stats.npy')
        #     self.hidden_size = self.hidden_size * 2

        self.fc1 = nn.Linear((self.nb_bins + self.nb_features) * self.nb_channels, self.hidden_size, bias=False)

        self.bn1 = nn.BatchNorm1d(self.hidden_size)

        if unidirectional:
            lstm_hidden_size = self.hidden_size
        else:
            lstm_hidden_size = self.hidden_size // 2

        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=nb_layers,
            bidirectional=not unidirectional,
            batch_first=False,
            dropout=0.4 if nb_layers > 1 else 0,
        )

        fc2_hiddensize = self.hidden_size * 2
        self.fc2 = nn.Linear(in_features=fc2_hiddensize, out_features=self.hidden_size, bias=False)

        self.bn2 = nn.BatchNorm1d(self.hidden_size)

        self.fc3 = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.nb_output_bins * self.nb_channels,
            bias=False,
        )

        self.bn3 = nn.BatchNorm1d(self.nb_output_bins * self.nb_channels)

        if input_mean is not None:
            if self.nb_bins != 0:
                input_mean = torch.from_numpy(-input_mean[: self.nb_bins]).float()
            else:
                input_mean = torch.from_numpy(-input_mean[: self.nb_features]).float()
        else:
            if self.nb_bins != 0:
                input_mean = torch.zeros(self.nb_bins)
            else:
                input_mean = torch.zeros(self.nb_features)
        if input_scale is not None:
            if self.nb_bins != 0:
                input_scale = torch.from_numpy(1.0 / input_scale[: self.nb_bins]).float()
            else:
                input_scale = torch.from_numpy(1.0 / input_scale[: self.nb_features]).float()
        else:
            if self.nb_bins != 0:
                input_scale = torch.ones(self.nb_bins)
            else:
                input_scale = torch.ones(self.nb_features)

        self.input_mean = nn.Parameter(input_mean)
        self.input_scale = nn.Parameter(input_scale)

        self.output_scale = nn.Parameter(torch.ones(self.nb_output_bins).float())
        self.output_mean = nn.Parameter(torch.ones(self.nb_output_bins).float())

        stft, _ = transforms.make_filterbanks(n_fft=self.nfft, n_hop=self.nhop, center=True, sample_rate=self.sample_rate)
        self.encoder = torch.nn.Sequential(stft, model.ComplexNorm(mono=self.nb_channels == 1))
    
    def get_loss(self):
        return nn.MSELoss()
    
    def forward(self, stft, x):

        # permute so that batch is last for lstm
        stft = stft.permute(3, 0, 1, 2)

        # get current spectrogram shape
        nb_frames, nb_samples, nb_channels, nb_bins = stft.data.shape

        mix = stft.detach().clone()

        x = x.mean(dim=1)
        input1 = self.bert.process_wav(x).to(x.device)  # [batch_size, seq_length]
        padding = torch.zeros(input1.shape[0], 320, device=input1.device)  # [batch_size, 2, hidden_dim]
        input1 = torch.cat((input1, padding), dim=1)     # [batch_size. 160400]

        if self.cfg.layer == "all":
            with torch.no_grad():
                num = input1.shape[1] // self.max_len + 1
                if num <= 1:
                    x = self.bert(input1, layer=None, reduction="none")[1:]  # [12, batch_size, seq_length, hidden_dim]
                else:
                    for i in range(num):
                        if i == 0:
                            x = self.bert(input1[:, :self.max_len], layer=None, reduction="none")[1:]
                        else:
                            x = torch.concat([x, self.bert(input1[:, i * self.max_len: (i + 1) * self.max_len], layer=None, reduction="none")[1:]], dim=2)
            x = (F.softmax(self.aggregator, dim=0) * x).sum(dim=0)  # [batch_size, seq_length, hidden_dim]
        else:
            with torch.no_grad():
                x = self.bert(input1, layer=int(self.cfg.layer), reduction="none")  # [batch_size, seq_length, hidden_dim]

        if x.shape[1] >= nb_frames:
            x = x[:, :nb_frames]
        else:
            pad_len = nb_frames - x.shape[1]
            x = F.pad(x, pad=(0, 0, 0, pad_len))
        x = x.unsqueeze(1).permute(2, 0, 1, 3)

        if self.nb_bins != 0:
            # crop
            stft = stft[..., : self.nb_bins]
            # shift and scale input to mean=0 std=1 (across all bins)
            stft = stft + self.input_mean
            stft = stft * self.input_scale
            x = torch.concat([x, stft], dim=-1)
        else:
            x = x + self.input_mean
            x = x * self.input_scale

        # to (nb_frames*nb_samples, nb_channels*nb_bins)
        # and encode to (nb_frames*nb_samples, hidden_size)
        x = self.fc1(x.reshape(-1, self.nb_channels * (self.nb_bins + self.nb_features)))
        # normalize every instance in a batch
        x = self.bn1(x)
        x = x.reshape(nb_frames, nb_samples, self.hidden_size)
        # squash range ot [-1, 1]
        x = torch.tanh(x)

        # apply 3-layers of stacked LSTM
        lstm_out = self.lstm(x)

        # lstm skip connection
        x = torch.cat([x, lstm_out[0]], -1)

        # first dense stage + batch norm
        x = self.fc2(x.reshape(-1, x.shape[-1]))
        x = self.bn2(x)

        x = F.relu(x)

        # second dense stage + layer norm
        x = self.fc3(x)
        x = self.bn3(x)

        # reshape back to original dim
        x = x.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)

        # apply output scaling
        x *= self.output_scale
        x += self.output_mean

        # since our output is non-negative, we can apply RELU
        x = F.relu(x) * mix
        # permute back to (nb_samples, nb_channels, nb_bins, nb_frames)
        return x.permute(1, 2, 3, 0)

    @torch.no_grad()
    def decoder(self, audio, spectrograms):
        nb_sources = self.nb_targets
        nb_samples = audio.shape[0]

        stft, istft = transforms.make_filterbanks(n_fft=self.nfft, n_hop=self.nhop, center=True, sample_rate=self.sample_rate)
        stft = stft.to(audio.device)
        istft = istft.to(audio.device)

        # getting the STFT of mix:
        # (nb_samples, nb_channels, nb_bins, nb_frames, 2)
        mix_stft = stft(audio)

        # transposing it as
        # (nb_samples, nb_frames, nb_bins,{1,nb_channels}, nb_sources)
        spectrograms = spectrograms.unsqueeze(-1).permute(0, 3, 2, 1, 4)

        # rearranging it into:
        # (nb_samples, nb_frames, nb_bins, nb_channels, 2) to feed
        # into filtering methods
        mix_stft = mix_stft.permute(0, 3, 2, 1, 4)

        # create an additional target if we need to build a residual
        if self.residual:
            # we add an additional target
            nb_sources += 1

        if nb_sources == 1 and self.niter > 0:
            raise Exception(
                "Cannot use EM if only one target is estimated."
                "Provide two targets or create an additional "
            )

        nb_frames = spectrograms.shape[1]
        targets_stft = torch.zeros(
            mix_stft.shape + (nb_sources,), dtype=audio.dtype, device=mix_stft.device
        )
        for sample in range(nb_samples):
            pos = 0
            if self.wiener_win_len:
                wiener_win_len = self.wiener_win_len
            else:
                wiener_win_len = nb_frames
            while pos < nb_frames:
                cur_frame = torch.arange(pos, min(nb_frames, pos + wiener_win_len))
                pos = int(cur_frame[-1]) + 1

                targets_stft[sample, cur_frame] = filtering.wiener(
                    spectrograms[sample, cur_frame],
                    mix_stft[sample, cur_frame],
                    self.niter,
                    softmask=self.softmask,
                    residual=self.residual,
                )

        # getting to (nb_samples, nb_targets, channel, fft_size, n_frames, 2)
        targets_stft = targets_stft.permute(0, 5, 3, 2, 1, 4).contiguous()

        # inverse STFT
        estimates = istft(targets_stft, length=audio.shape[2])

        return estimates[:, :-1].squeeze(1)
    
    def init_metrics(self):
        self.all_metrics = set()

        for split in ['train', 'valid', 'test']:
            setattr(self, f"{split}_sdr", GlobalSDR())
            self.all_metrics.add('sdr')
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        X = self.encoder(x)
        Y = self.encoder(y)
        Y_pred = self(X, x)
        y_pred = self.decoder(x, Y_pred)
        loss = self.loss(Y_pred, Y)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        # self.update_metrics('train', y, y_pred)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        X = self.encoder(x)
        Y = self.encoder(y)
        Y_pred = self(X, x)
        y_pred = self.decoder(x, Y_pred)
        loss = self.loss(Y_pred, Y)
        self.log('valid_loss', loss, prog_bar=True, sync_dist=True)
        self.update_metrics('valid', y, y_pred)

    def validation_epoch_end(self, outputs):
        self.log_metrics('valid')

    def test_step(self, batch, batch_idx):
        x, y = batch
        X = self.encoder(x)
        Y = self.encoder(y)
        Y_pred = self(X, x)
        y_pred = self.decoder(x, Y_pred)
        loss = self.loss(Y_pred, Y)
        self.log('test_loss', loss, prog_bar=True, sync_dist=True)
        self.update_metrics('test', y, y_pred)
    
    def test_epoch_end(self, outputs):
        self.log_metrics('test')
        if self.cfg.layer == "all":
            if not isinstance(self.aggregator, nn.Conv1d):
                log_dict = self.log_weights('test')
                self.log_dict(log_dict)
    
    @torch.no_grad()
    def update_metrics(self, split, y=None, y_pred=None):  
        for metric_name in self.all_metrics:
            metric = getattr(self, f"{split}_{metric_name}")
            if len(y_pred.shape) == 2:
                separations = y_pred.unsqueeze(0)
                references = y.unsqueeze(0)
            else:
                separations = y_pred
                references = y
            metric.update(separations, references)

    @torch.no_grad()
    def log_metrics(self, split):
        self.log(f"{split}_sdr", getattr(self, f"{split}_sdr").compute(), sync_dist=True)
        getattr(self, f"{split}_sdr").reset()

    
    