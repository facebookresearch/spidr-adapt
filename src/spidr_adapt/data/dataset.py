# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Speech dataset and batch sampler. Adapted from torchaudio."""

import abc
import functools
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

import polars as pl
import torch
from torch import Tensor
from torch import distributed as dist
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import BatchSampler, DataLoader, Dataset, DistributedSampler
from torchcodec.decoders import AudioDecoder

from spidr_adapt.config import (
    ALIGNMENT_FREQ,
    DEFAULT_CONV_LAYER_CONFIG,
    SAMPLE_RATE,
    Config,
    DataConfig,
    MaskingConfig,
)
from spidr_adapt.data.masks import MaskGenerator
from spidr_adapt.data.tokenizer import Tokenizer, get_tokenizer_for_lang
from spidr_adapt.data.utils import (
    bytes_from_archive,
    divide_manifest_language_by_chunks,
    read_alignments,
    read_manifest,
)
from spidr_adapt.environment import pg_ddp, pg_metalearning

logger = logging.getLogger()


class BatchType(Enum):
    SSL = 1
    SUPERVISED = 2


def verify_lengths(min_len: int, max_len: int, batch_size: int | None, max_token_count: int | None) -> None:
    exception = ""
    if not 0 <= min_len <= max_len:
        exception += "min_len must be less than or equal to max_len \n"
    if max_token_count is not None and batch_size is not None:
        exception += "max_token_count and batch_size cannot be set simultaneously \n"
    if max_token_count is None and batch_size is None:
        exception += "max_token_count or batch_size must be set \n"
    if max_token_count is not None and max_len > max_token_count:
        exception += "max_len must be less than or equal to max_token_count \n"
    if exception:
        raise ValueError(exception)


def get_buckets(lengths: list[int], num_buckets: int, uniform_limits: tuple[int, int] | None) -> dict[int, Tensor]:
    buckets: dict[int, list[int]] = {}
    if uniform_limits is not None:
        boundaries = torch.linspace(uniform_limits[0] - 1, uniform_limits[1] + 1, num_buckets + 1)
    else:
        boundaries = torch.quantile(
            torch.tensor(lengths, dtype=torch.float32),
            torch.linspace(0, 1, num_buckets + 1),
            interpolation="lower",
        )[1:]
    bucket_ids = torch.bucketize(torch.tensor(lengths), boundaries)
    for i in range(bucket_ids.size(0)):
        bucket_id = int(bucket_ids[i])
        if bucket_id in buckets:
            buckets[bucket_id].append(i)
        else:
            buckets[bucket_id] = [i]
    return dict(sorted([(k, torch.as_tensor(v, dtype=torch.int)) for k, v in buckets.items()]))


@dataclass(frozen=True)
class SampleBatch:
    waveforms: Tensor
    phonemes: Tensor | None = None
    languages: list[str | None] | None = None
    attention_mask: Tensor | None = None
    mask: Tensor | None = None

    def to(self, device: torch.device) -> "SampleBatch":
        return SampleBatch(
            waveforms=self.waveforms.to(device),
            phonemes=self.phonemes.to(device) if self.phonemes is not None else None,
            languages=self.languages,
            attention_mask=self.attention_mask.to(device) if self.attention_mask is not None else None,
            mask=self.mask.to(device) if self.mask is not None else None,
        )


class BucketizeBatchSampler(BatchSampler):
    """Batch sampler that groups samples of similar length into buckets."""

    def __init__(
        self,
        *,
        lengths: dict[int, int],
        num_buckets: int,
        min_len: int,
        max_len: int | None,
        max_token_count: int | None,
        batch_size: int | None,
        seed: int,
        bucket_method: Literal["uniform", "percentile"],
        shuffle: bool,
        drop_last: bool,
    ) -> None:
        if max_len is None:
            max_len = max(lengths)
        verify_lengths(min_len, max_len, batch_size, max_token_count)
        filtered_length_idx = [(min(length, max_len), i) for i, length in lengths.items() if min_len <= length]
        if not filtered_length_idx:
            exception = "No samples with length in the range"
            raise ValueError(exception)

        sorted_filtered_length_idx = sorted(filtered_length_idx, key=lambda x: x[0])
        self.lengths = [e[0] for e in sorted_filtered_length_idx]
        self.indices = [e[1] for e in sorted_filtered_length_idx]
        self.max_token_count = max_token_count
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        if self.shuffle:
            self.generator = torch.Generator().manual_seed(self.seed)
        self.drop_last = drop_last
        if bucket_method == "uniform":
            uniform_limits = (min_len, max_len)
        elif bucket_method == "percentile":
            uniform_limits = None
        else:
            raise ValueError("bucket_method must be either 'uniform' or 'percentile'")
        self.buckets = get_buckets(self.lengths, num_buckets, uniform_limits)
        self._update_iter_list()

    def _update_iter_list(self) -> None:
        if self.shuffle:
            for k in self.buckets:
                new_idx = torch.randperm(self.buckets[k].size(0), generator=self.generator)
                self.buckets[k] = self.buckets[k][new_idx]
        self.iter_list = []
        total_len, batch = 0, []
        max_batch_size = self.max_token_count or self.batch_size
        for k in self.buckets:
            for i in range(self.buckets[k].size(0)):
                index = int(self.buckets[k][i])
                sample_length = self.lengths[index] if self.max_token_count else 1
                if total_len + sample_length <= max_batch_size:
                    batch.append(self.indices[index])
                    total_len += sample_length
                else:
                    self.iter_list.append(batch)
                    batch = [self.indices[index]]
                    total_len = sample_length
        if len(batch) > 0 and (self.max_token_count or not self.drop_last):
            self.iter_list.append(batch)

    def set_epoch(self, epoch: int) -> None:
        self.seed += epoch
        self.generator.manual_seed(self.seed)
        self._update_iter_list()

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self.iter_list)

    def __len__(self) -> int:
        return len(self.iter_list)


class DistributedBatchSampler(DistributedSampler):
    """Distributed sampler for BucketizeBatchSampler."""

    def __init__(
        self, batch_samplers: dict[str, BucketizeBatchSampler], *, seed: int, shuffle: bool, drop_last: bool
    ) -> None:
        self.batch_samplers = batch_samplers
        self.num_replicas = dist.get_world_size(pg_ddp()) if dist.is_initialized() else 1
        self.rank = dist.get_rank(pg_ddp()) if dist.is_initialized() else 0
        self.metalearning_rank = dist.get_rank(pg_metalearning()) if dist.is_initialized() else 0
        self.metalearning_world_size = dist.get_world_size(pg_metalearning()) if dist.is_initialized() else 1
        self.shuffle = shuffle
        self.epoch = 0
        self.seed = seed
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.batch_language = None
        self.batch_language_list = None

    def set_epoch(self, epoch: int) -> None:
        for key in self.batch_samplers:
            self.batch_samplers[key].set_epoch(epoch)
        return super().set_epoch(epoch)

    def set_batch_language_task(
        self,
        step: int,
        reset_interval: int,
        fixed_language: str | None = None,
    ) -> bool:
        if fixed_language:
            self.batch_language = fixed_language
        else:
            num_reset = step // reset_interval
            if not self.batch_language_list:
                g = torch.Generator()
                g.manual_seed(self.seed)
                languages = list(self.batch_samplers.keys())
                shuffled_indices = torch.randperm(len(languages), generator=g)
                self.batch_language_list = [languages[i] for i in shuffled_indices]
            batch_language_index = (num_reset * self.metalearning_world_size + self.metalearning_rank) % len(
                self.batch_language_list
            )  # each worker on each reset should have a unique index
            self.batch_language = self.batch_language_list[batch_language_index]
        logger.info("Setting batch language task to %s", self.batch_language)

    def __iter__(self) -> Iterator[list[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch + self.metalearning_rank)
        subsets = []
        language_batches = (
            [self.batch_samplers[self.batch_language]] if self.batch_language else self.batch_samplers.values()
        )
        for batch_sampler in language_batches:
            if self.shuffle:  # First, shuffle each sampler independently
                perm = torch.randperm(len(batch_sampler.iter_list), generator=g).tolist()
                indices = [batch_sampler.iter_list[i] for i in perm]
            else:
                indices = batch_sampler.iter_list
            if self.drop_last:
                total_size = len(indices) - len(indices) % self.num_replicas
            else:
                padding_size = self.num_replicas - len(indices) % self.num_replicas
                indices += indices[:padding_size]
                total_size = len(indices)
            num_samples = total_size // self.num_replicas
            subset = indices[self.rank : total_size : self.num_replicas]
            if len(subset) != num_samples:
                exception = f"Rank {self.rank} has subset of length {len(subset)} but expected {num_samples}"
                raise ValueError(exception)
            subsets += subset

        self.num_samples = len(subsets)
        g = torch.Generator().manual_seed(self.seed + self.epoch + self.metalearning_rank)
        if len(self.batch_samplers) > 1 and self.shuffle:  # Now shuffle across samplers
            subsets = [subsets[i] for i in torch.randperm(self.num_samples, generator=g).tolist()]
        self.subset = subsets
        return iter(self.subset)

    def __len__(self) -> int:
        return self.num_samples


class SpeechDataset(Dataset, abc.ABC):
    """Dataset to load chunks of audio files."""

    def __init__(
        self,
        manifest_path: Path | str,
        *,
        normalize: bool,
        alignments_path: Path | str | None = None,
        lang_task_chunk_duration: int | None = None,
        random_seed: int = 0,
    ) -> None:
        super().__init__()
        self.manifest = read_manifest(manifest_path)
        if lang_task_chunk_duration is not None:
            self.manifest = divide_manifest_language_by_chunks(
                self.manifest, SAMPLE_RATE, lang_task_chunk_duration, seed=random_seed
            )
        self.normalize = normalize
        self.phoneme_tokens = None
        if alignments_path:
            alignments = read_alignments(alignments_path)
            self.tokenizers = {}
            self.phoneme_tokens = {
                fileid: get_tokenizer_for_lang(self.tokenizers, language).encode(phonemes)
                for fileid, (phonemes, language) in alignments.items()
            }
        self.lengths = list(self.manifest["num_frames"])

    def __len__(self) -> int:
        return len(self.manifest)

    @abc.abstractmethod
    def _load_audio(self, index: int) -> tuple[Tensor, int]:
        pass

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor | None, str | None]:
        waveform, sr = self._load_audio(index)
        if sr != SAMPLE_RATE or waveform.shape[0] != 1:
            raise ValueError(index)
        if self.normalize:
            waveform = F.layer_norm(waveform, waveform.shape)
        tokens = self.phoneme_tokens[self.manifest[index, "fileid"]] if self.phoneme_tokens is not None else None
        language = self.manifest[index, "language"]
        return waveform.squeeze(), tokens, language


class SpeechDatasetFromArchive(SpeechDataset):
    def _load_audio(self, index: int) -> tuple[Tensor, int]:
        entry = self.manifest[index].to_dicts()[0]
        data = bytes_from_archive(entry["archive"], entry["byte_offset"], entry["byte_size"])
        samples = AudioDecoder(data).get_all_samples()
        return samples.data, samples.sample_rate


class SpeechDatasetFromFiles(SpeechDataset):
    def _load_audio(self, index: int) -> tuple[Tensor, int]:
        samples = AudioDecoder(self.manifest[index, "path"]).get_all_samples()
        return samples.data, samples.sample_rate


def speech_dataset(
    manifest_path: Path | str,
    *,
    normalize: bool,
    alignments_path: Path | str | None = None,
    lang_task_chunk_duration: int | None = None,
    random_seed: int = 0,
) -> SpeechDataset:
    with Path(manifest_path).open("r", encoding="utf-8") as f:
        columns = set(f.readline().strip().split(","))
    if {"fileid", "path", "num_samples", "archive", "byte_offset", "byte_size"}.issubset(columns):
        return SpeechDatasetFromArchive(manifest_path, normalize=normalize, alignments_path=alignments_path)
    return SpeechDatasetFromFiles(
        manifest_path,
        normalize=normalize,
        alignments_path=alignments_path,
        lang_task_chunk_duration=lang_task_chunk_duration,
        random_seed=random_seed,
    )


def conv_length(shapes: list[tuple[int, int, int]], length: Tensor) -> Tensor:
    for _, kernel_size, stride in shapes:
        length = torch.div(length - kernel_size, stride, rounding_mode="floor") + 1
        length = torch.max(torch.zeros_like(length), length)
    return length


def build_padding_mask(lengths: torch.Tensor) -> torch.Tensor:
    batch_size, max_len = lengths.size(0), int(lengths.max())
    return torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) >= lengths[:, None]


def build_attention_mask(lengths: torch.Tensor) -> torch.Tensor:
    batch_size, max_len = lengths.size(0), int(lengths.max())
    padding_mask = build_padding_mask(lengths)
    return ~padding_mask[:, None, None, :].expand(batch_size, 1, max_len, max_len)


class SpeechCollatorWithMasking:
    def __init__(
        self,
        mask_generator: MaskGenerator,
        *,
        max_sample_size: int,
        conv_layer_config: list[tuple[int, int, int]],
        enable_padding: bool,
        rand_crop: bool,
        collate_phonemes: bool,
    ) -> None:
        self.mask_generator = mask_generator
        self.max_sample_size = max_sample_size
        self.conv_layer_config = conv_layer_config
        self.enable_padding = enable_padding
        self.rand_crop = rand_crop
        self.collate_phonemes = collate_phonemes

    def process_phoneme_tokens(self, tokens: Tensor, frame_offset: int, num_samples: int) -> Tensor:
        model_downsampling = functools.reduce(lambda x, y: x * y, [layer[2] for layer in self.conv_layer_config])
        model_freq = SAMPLE_RATE // model_downsampling  # in Hz
        alignment_downsampling = SAMPLE_RATE // ALIGNMENT_FREQ

        phoneme_frame_offset = frame_offset // alignment_downsampling
        phoneme_num_samples = num_samples // alignment_downsampling
        tokens = tokens[phoneme_frame_offset : phoneme_frame_offset + phoneme_num_samples]

        subsample = ALIGNMENT_FREQ // model_freq
        subsampled = tokens[::subsample]
        output_length = conv_length(self.conv_layer_config, num_samples)
        if len(subsampled) == output_length:
            return subsampled
        if len(subsampled) == output_length + 1:
            return subsampled[:-1]
        raise ValueError(f"Length mismatch: {len(tokens)} vs {output_length}")

    def crop_samples(
        self, sample: tuple[Tensor, Tensor | None, str | None], num_frames: int
    ) -> tuple[Tensor, Tensor | None, int]:
        frame_offset = 0
        waveform, phonemes, _ = sample

        wav_length = waveform.size(0)
        num_frames = min(num_frames, self.max_sample_size)
        if wav_length > num_frames and self.rand_crop:
            frame_offset = int(torch.randint(wav_length - num_frames, size=(1,)))
        elif wav_length < num_frames:
            num_frames = wav_length

        phonemes = self.process_phoneme_tokens(phonemes, frame_offset, num_frames) if self.collate_phonemes else None

        return waveform[frame_offset : frame_offset + num_frames], phonemes, num_frames

    def __call__(self, batch: list[tuple[Tensor, Tensor | None, str | None]]) -> SampleBatch:
        wavs, _, languages = zip(*batch, strict=True)
        num_frames = max(len(wav) for wav in wavs) if self.enable_padding else min(len(wav) for wav in wavs)
        wav_list, phoneme_list, wav_lengths = zip(
            *[self.crop_samples(data_item, num_frames) for data_item in batch], strict=True
        )
        waveforms = pad_sequence(wav_list, batch_first=True)  # type: ignore[arg-type]
        padded_phonemes = None
        if self.collate_phonemes:
            language_set = set(languages)
            assert len(language_set) == 1, "Please perform language batching if collating phonemes."
            language = language_set.pop()
            padded_phonemes = pad_sequence(phoneme_list, batch_first=True, padding_value=Tokenizer(language).ignore_id)

        lengths = conv_length(self.conv_layer_config, torch.tensor(wav_lengths))
        padding_mask, attention_mask = build_padding_mask(lengths), build_attention_mask(lengths)
        mask_indices: Tensor | None = self.mask_generator(padding_mask)[0]
        return SampleBatch(
            waveforms=waveforms,
            phonemes=padded_phonemes,
            languages=languages,
            attention_mask=attention_mask,
            mask=mask_indices,
        )


def lengths_by_lang(dataset: SpeechDataset) -> dict[str, dict[int, int]]:
    return {
        str(lang): dict(zip(df["index"].to_list(), df["num_frames"].to_list(), strict=True))
        for (lang,), df in (
            pl.from_dataframe(dataset.manifest)
            .select("language", "num_frames")
            .with_row_index()
            .group_by("language", maintain_order=True)
        )
    }


def build_dataloader(
    data_cfg: DataConfig,
    mask_cfg: MaskingConfig,
    conv_layer_config: list[tuple[int, int, int]],
) -> DataLoader:
    collate_fn = SpeechCollatorWithMasking(
        MaskGenerator(mask_cfg),
        max_sample_size=data_cfg.max_sample_size,
        conv_layer_config=conv_layer_config or DEFAULT_CONV_LAYER_CONFIG,
        enable_padding=data_cfg.enable_padding,
        rand_crop=data_cfg.rand_crop,
        collate_phonemes=bool(data_cfg.alignments_path),
    )

    dataset = speech_dataset(
        data_cfg.manifest,
        normalize=data_cfg.normalize,
        alignments_path=data_cfg.alignments_path,
        lang_task_chunk_duration=data_cfg.lang_task_chunk_duration,
        random_seed=data_cfg.random_seed,
    )
    if data_cfg.by_lang:
        batch_samplers = {
            lang: BucketizeBatchSampler(
                lengths=lengths,
                num_buckets=data_cfg.num_buckets,
                min_len=data_cfg.min_sample_size,
                max_len=data_cfg.max_sample_size,
                max_token_count=data_cfg.max_batch_length,
                batch_size=None,
                seed=data_cfg.random_seed,
                bucket_method=data_cfg.bucket_method,
                shuffle=True,
                drop_last=data_cfg.drop_last,
            )
            for lang, lengths in lengths_by_lang(dataset).items()
        }
    else:
        batch_samplers = {
            "any": BucketizeBatchSampler(
                lengths=dict(zip(range(len(dataset.lengths)), dataset.lengths, strict=True)),
                num_buckets=data_cfg.num_buckets,
                min_len=data_cfg.min_sample_size,
                max_len=data_cfg.max_sample_size,
                max_token_count=data_cfg.max_batch_length,
                batch_size=None,
                seed=data_cfg.random_seed,
                bucket_method=data_cfg.bucket_method,
                shuffle=True,
                drop_last=data_cfg.drop_last,
            )
        }
    distributed_sampler = DistributedBatchSampler(
        batch_samplers,
        seed=data_cfg.random_seed,
        shuffle=True,
        drop_last=data_cfg.drop_last,
    )
    return DataLoader(
        dataset,
        batch_sampler=distributed_sampler,
        num_workers=data_cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=data_cfg.pin_memory,
        prefetch_factor=data_cfg.prefetch_factor,
        persistent_workers=data_cfg.persistent_workers,
        generator=torch.Generator().manual_seed(
            data_cfg.random_seed + (dist.get_rank() if dist.is_initialized() else 0)
        ),
    )


class InterleaveSLDatasetLoader:
    def __init__(self, cfg: Config, epoch: int, *, supervised_epoch: int = 0) -> None:
        self.cfg = cfg

        ssl_data_cfg = cfg.data if isinstance(cfg.data, DataConfig) else cfg.data[BatchType.SSL.name.lower()]
        self.ssl_loader, self.ssl_loader_iter = self.initialize_loaders(ssl_data_cfg)
        self.epoch = epoch
        self.start_new_epoch(BatchType.SSL)

        if cfg.model.supervised_every_step:
            supervised_data_cfg = (
                cfg.data.get(BatchType.SUPERVISED.name.lower(), None) if isinstance(cfg.data, dict) else None
            )
            assert supervised_data_cfg, "Supervised interleaved training requires supervised data in the config."
            self.supervised_loader, self.supervised_loader_iter = self.initialize_loaders(supervised_data_cfg)
            self.supervised_epoch = supervised_epoch
            self.start_new_epoch(BatchType.SUPERVISED)

    def initialize_loaders(self, data_cfg: DataConfig) -> tuple[DataLoader, Iterator]:
        loader = build_dataloader(data_cfg, self.cfg.masking, self.cfg.model.extractor_conv_layer_config)
        return loader, iter(loader)

    def start_new_epoch(self, batch_type: BatchType) -> tuple[Iterator]:
        if batch_type == BatchType.SSL:
            self.epoch += 1
            self.ssl_loader.batch_sampler.set_epoch(self.epoch)
            self.ssl_loader_iter = iter(self.ssl_loader)
            logger.info("Starting ssl epoch %s", int(self.epoch))
        elif batch_type == BatchType.SUPERVISED:
            self.supervised_epoch += 1
            self.supervised_loader.batch_sampler.set_epoch(self.supervised_epoch)
            self.supervised_loader_iter = iter(self.supervised_loader)
            logger.info("Starting supervised epoch %s", int(self.supervised_epoch))
        else:
            raise ValueError(f"Unknown data type: {batch_type}")

    def get_next_batch(
        self, loader_iterator: Iterator
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            return next(loader_iterator)
        except StopIteration:
            self.start_new_epoch(BatchType.SSL if loader_iterator is self.ssl_loader_iter else BatchType.SUPERVISED)
            return next(loader_iterator)

    def load_batch_data(self, batch_type: BatchType) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if batch_type == BatchType.SSL:
            return self.get_next_batch(self.ssl_loader_iter)
        if batch_type == BatchType.SUPERVISED:
            return self.get_next_batch(self.supervised_loader_iter)
        raise ValueError(f"Unknown data type: {batch_type}")

    def set_task(self, step: int, reset_interval: int) -> None:
        self.ssl_loader.batch_sampler.set_batch_language_task(step, reset_interval)
        self.ssl_loader_iter = iter(self.ssl_loader)
        batch_language = self.ssl_loader.batch_sampler.batch_language.split("_chunk")[0]
        if self.cfg.model.supervised_every_step:
            self.supervised_loader.batch_sampler.set_batch_language_task(
                step, reset_interval, fixed_language=batch_language
            )
            self.supervised_loader_iter = iter(self.supervised_loader)
