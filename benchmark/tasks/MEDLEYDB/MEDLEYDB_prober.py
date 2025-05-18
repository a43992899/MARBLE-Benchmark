import torch
import torch.nn.functional as F

from torch import nn

import benchmark as bench
from benchmark.tasks.MEDLEYDB.MEDLEYDB_metric import MEDLEYDBMeasure
from benchmark.tasks.MEDLEYDB.MEDLEYDB_dataset import MEDLEYDBAudioDataset


class MEDLEYDBProber(bench.ProberForBertSeqLabel):
    """MLP Prober for sequence labeling tasks with BERT like 12 layer features (HuBERT, Data2vec).

    Sequence labeling tasks are token level classification tasks, in MIR, it is usually used for
    onset detection, note transcription, etc.

    This class supports learnable weighted sum over different layers of features.

    TODO: support official evaluation strategy
    TODO: enable biLSTM
    TODO: fix refresh rate
    """

    def __init__(self, cfg):
        self.pretrain_output = cfg.model.feature_extractor.pretrain.num_features
        cfg.model.feature_extractor.pretrain.num_features = 2 * self.pretrain_output
        super().__init__(cfg)
        self.voicing_threshold = cfg.model.voicing_threshold
        self.post_processing = getattr(self, cfg.model.post_processing)
        self.lstm = nn.LSTM(
            input_size=self.pretrain_output,
            hidden_size=self.pretrain_output,
            num_layers=1,
            bidirectional=True,
            dropout=self.hparams.dropout_p,
        )
        self.batch_norm = nn.BatchNorm1d(self.hidden_layer_sizes[-1])

    def forward(self, x):
        padding = torch.zeros(
            x.shape[0], 320, device=x.device
        )  # this is not common for every model
        x = torch.cat((x, padding), dim=1)
        x = self.bert.process_wav(x).to(x.device)  # [B, T]
        if self.hparams.layer == "all":
            with torch.no_grad():
                x = self.bert(x, layer=None, reduction="none")[
                    1:
                ]  # [12, batch_size, seq_length (375), hidden_dim]
            x = (F.softmax(self.aggregator, dim=0) * x).sum(
                dim=0
            )  # [batch_size, seq_length (375), hidden_dim]
        else:
            with torch.no_grad():
                x = self.bert(
                    x, layer=int(self.hparams.layer), reduction="none"
                )  # [batch_size, seq_length (375), hidden_dim]

        batch_size, clip_size, _ = x.shape
        x = self.lstm(x)[0]  # batch_size, clip_size, hidden_dim
        x = x.flatten(0, 1)
        for i in range(self.num_layers):
            x = getattr(self, f"hidden_{i}")(x)
            x = self.batch_norm(x)
            x = F.relu(x)

        output = self.output(x)
        return output.view(batch_size, clip_size, -1)

    def training_step(self, batch, batch_idx):
        clips, offsets, labels = batch
        y_pred = self(clips)
        loss = self.loss(y_pred, self._align_labels(labels, y_pred.shape[1], offsets))
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.update_metrics("train", batch[1:], y_pred)
        return loss

    def validation_step(self, batch, batch_idx):
        clips, offsets, labels = batch  # x: [batch_size, n_layer_features, hidden_dim]
        y_pred = self(clips)  # [batch_size, n_class]
        loss = self.loss(y_pred, self._align_labels(labels, y_pred.shape[1], offsets))
        self.log(
            "valid_loss", loss, batch_size=clips.shape[0], prog_bar=True, sync_dist=True
        )
        self.update_metrics("valid", batch[1:], y_pred)

    def test_step(self, batch, batch_idx):
        clips, offsets, labels = batch
        y_pred = self(clips)  # [batch_size, n_class]
        loss = self.loss(y_pred, self._align_labels(labels, y_pred.shape[1], offsets))
        self.log(
            "test_loss", loss, batch_size=clips.shape[0], prog_bar=True, sync_dist=True
        )
        self.update_metrics("test", batch[1:], y_pred)

    def init_metrics(self):
        self.all_metrics = set()

        for split in ["train", "valid", "test"]:
            setattr(self, f"{split}_oas", MEDLEYDBMeasure())
            self.all_metrics.add("oas")

    @torch.no_grad()
    def update_metrics(self, split, y, pred):
        """update metrics during train/valid/test step"""
        pred = pred.detach()
        pred = torch.sigmoid(pred)

        offsets, labels = y
        valid_indices = labels[..., 0] > 1e-6
        assert (
            pred.shape[-1] == 360
        ), f"Predictions have {pred.shape[-1]} classes, should be 360."
        # est_notes: B x L'
        peak_prob, peak_idx = pred.max(dim=-1)
        est_notes = self.est_notes_fn(
            peak_idx, pred, peak_prob < self.voicing_threshold
        )
        est_freqs = self.est_notes_to_hz_fn(est_notes)
        # No longer assumes estimating melody has the same resolution as the target.
        # assert est_freqs.shape == freqs.shape
        est_times = torch.linspace(
            0,
            pred.shape[1] / self.hparams.token_rate,
            pred.shape[1],
            device=pred.device,
        )
        est_times = offsets[:, None] + est_times
        # time value of 0 are paddings and should not be used in evaluation.
        targets = [
            labels[i, valid_idx, :2] for i, valid_idx in enumerate(valid_indices)
        ]
        ref_times = [target[:, 0] for target in targets]
        est_freqs = self.post_processing(est_times, est_freqs, ref_times)
        predictions = [
            torch.stack([ref_time, est_freq], dim=-1)
            for ref_time, est_freq in zip(ref_times, est_freqs)
        ]
        # predictions = torch.stack([targets[:, 0], est_freqs], dim=-1)
        getattr(self, f"{split}_oas").update(predictions, targets)

    @torch.no_grad()
    def log_metrics(self, split):
        self.log(
            f"{split}_oas", getattr(self, f"{split}_oas").compute(), sync_dist=True
        )
        getattr(self, f"{split}_oas").reset()

    @torch.no_grad()
    def _align_labels(self, labels, token_size, offsets):
        token_rate = self.hparams.token_rate
        feature_times = torch.linspace(
            0, token_size / token_rate, token_size, device=offsets.device
        )
        feature_times = feature_times[:, None] + offsets
        nearests = (
            torch.abs((feature_times[..., None] - labels[..., 0])).argmin(dim=-1).T
        )
        aligned_labels = torch.gather(labels[..., -1], 1, nearests)
        return self.one_hot(aligned_labels)

    @staticmethod
    def one_hot(notes):
        note_range = MEDLEYDBAudioDataset.note_range.to(notes.device)
        min_note = note_range[0] - 0.4
        notes = torch.maximum(
            notes, torch.full(notes.shape, min_note, device=notes.device)
        )

        max_note = note_range[-1] + 0.4
        # max_note = max_note.astype(np.float32)
        notes = torch.minimum(
            notes, torch.full(notes.shape, max_note, device=notes.device)
        )

        notes = notes[..., None] - note_range[None, :]
        notes = -(notes**2) / (2.0 * 0.18**2)
        notes = torch.exp(notes)
        notes = torch.where(
            notes < 4e-3, torch.zeros_like(notes, device=notes.device), notes
        )
        return notes

    @staticmethod
    def est_notes_to_hz_fn(est_notes, ref_freq=440):
        min_note = MEDLEYDBAudioDataset.note_range[0]
        positive_idx = est_notes >= min_note
        negative_idx = est_notes <= -min_note
        est_freqs_with_voicing = torch.empty_like(est_notes)
        est_freqs_with_voicing[positive_idx] = ref_freq * (
            2 ** ((est_notes[positive_idx] - 69) / 12)
        )
        est_freqs_with_voicing[negative_idx] = -ref_freq * (
            2 ** ((-est_notes[negative_idx] - 69) / 12)
        )
        return est_freqs_with_voicing

    @staticmethod
    def est_notes_fn(est_peak_indices, est_probs, est_voicing):
        """
        Get weighted melody notes from predictions.
        Reference: https://github.com/drwangxian/dilated_conv_model_for_melody_extraction
        """
        batch_size, frame_size = est_peak_indices.shape
        est_peak_indices, est_probs = est_peak_indices.flatten(), est_probs.flatten(
            0, 1
        )
        note_range = MEDLEYDBAudioDataset.note_range.to(est_probs.device)
        note_offset = note_range[0]
        note_range = (note_range - note_offset).to(torch.float32)

        frames_360 = torch.arange(360, dtype=torch.int32, device=est_probs.device)
        peak_masks = est_peak_indices[:, None] - frames_360[None, :]
        peak_masks = peak_masks.view(-1, 360)
        peak_masks = torch.abs(peak_masks) <= 1
        masked_probs = torch.where(
            peak_masks, est_probs, torch.zeros_like(est_probs, device=est_probs.device)
        ).view(-1, 360)
        normalization_probs = torch.sum(masked_probs, dim=1).view(-1)
        frames_72 = note_range
        est_notes = frames_72[None, :] * masked_probs
        est_notes = torch.sum(est_notes, dim=1).view(-1)
        est_notes = est_notes / torch.maximum(
            1e-3 * torch.ones_like(normalization_probs, device=est_probs.device),
            normalization_probs,
        )
        est_notes = est_notes + note_offset
        est_notes = est_notes.view(batch_size, frame_size)
        est_notes[est_voicing] *= -1
        return est_notes

    @staticmethod
    def nearest_interpolation(orig_times, orig_freqs, new_times):
        # assumes orig_freqs of shape (B, T)
        new_freqs = []
        for orig_freq, new_time in zip(orig_freqs, new_times):
            new_freq = F.interpolate(orig_freq[None, None, :], len(new_time)).squeeze()
            new_freq = torch.atleast_1d(new_freq)
            new_freqs.append(new_freq)
        return new_freqs
