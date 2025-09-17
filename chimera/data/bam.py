import multiprocessing
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import Any

import pysam
from datasets import Dataset as HuggingFaceDataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from chimera.data.tokenizer import (
    ID_FEATURE,
    QUAL_FEATURE,
    SEQ_FEATURE,
    DataCollator,
    tokenize_and_align_labels_and_quals,
    tokenize_and_align_labels_and_quals_ids,
)


def is_chimeric(read: pysam.AlignedSegment) -> bool:
    """Check if the read is chimeric."""
    return not read.is_unmapped and read.has_tag("SA") and not read.is_secondary and not read.is_supplementary


def parse_bam_file(file_path: Path) -> Iterator[dict]:
    """Parse a BAM file and extrand read with sa tag."""
    bam = pysam.AlignmentFile(file_path.as_posix(), "rb")

    for read in bam:
        if is_chimeric(read):
            yield {
                "id": read.query_name,
                "seq": read.query_sequence,
            }

    bam.close()


class BamDataModule(LightningDataModule):
    """`LightningDataModule` for the bam dataset.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```
    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        tokenizer,
        train_data_path: Path,
        batch_size: int,
        val_data_path: Path | None = None,
        test_data_path: Path | None = None,
        predict_data_path: Path | None = None,
        num_workers: int = 0,
        max_train_samples: int | None = None,
        max_val_samples: int | None = None,
        max_test_samples: int | None = None,
        max_predict_samples: int | None = None,
        *,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `BamDataModule`.

        :param batch_size: The batch size.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None
        self.batch_size_per_device = batch_size
        self.data_collator = DataCollator(tokenizer)

    @property
    def num_classes(self) -> int:
        """Get the number of classes."""
        return 2

    def prepare_data(self) -> None:
        """Encode the BAM data to Parquet format."""
        data_paths = [self.hparams.train_data_path]

        if self.hparams.val_data_path is not None:
            data_paths.append(self.hparams.val_data_path)

        if self.hparams.test_data_path is not None:
            data_paths.append(self.hparams.test_data_path)

        if self.hparams.predict_data_path is not None:
            data_paths.append(self.hparams.predict_data_path)

    def setup(self, stage: str | None = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                msg = f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                raise RuntimeError(msg)
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if stage == "predict":
            if not self.hparams.predict_data_path:
                msg = "Predict data path is required for prediction stage."
                raise ValueError(msg)

            num_proc = min(self.hparams.num_workers, multiprocessing.cpu_count() - 1)

            predict_dataset = HuggingFaceDataset.from_generator(
                parse_bam_file,
                gen_kwargs={"file_path": self.hparams.predict_data_path},
                num_proc=max(1, num_proc),
            ).with_format("torch")

            if self.hparams.max_predict_samples is not None:
                max_predict_samples = min(self.hparams.max_predict_samples, len(predict_dataset))
                predict_dataset = HuggingFaceDataset.from_dict(predict_dataset[:max_predict_samples]).with_format(
                    "torch"
                )

            self.data_predict = predict_dataset.map(
                partial(
                    tokenize_and_align_labels_and_quals_ids,
                    tokenizer=self.hparams.tokenizer,
                    max_length=self.hparams.tokenizer.max_len_single_sentence,
                ),
                num_proc=max(1, num_proc),  # type: ignore
            ).remove_columns([SEQ_FEATURE, QUAL_FEATURE])
            del predict_dataset
            return

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            num_proc = min(self.hparams.num_workers, multiprocessing.cpu_count() - 1)

            if (
                self.hparams.val_data_path is None
                or self.hparams.test_data_path is None
                or self.hparams.train_data_path is None
            ):
                msg = "Val, test, and train data paths are required for training."
                raise ValueError(msg)

            train_dataset = HuggingFaceDataset.from_generator(
                parse_bam_file,
                gen_kwargs={"file_path": self.hparams.train_data_path},
                num_proc=max(1, num_proc),
            ).with_format("torch")

            val_dataset = HuggingFaceDataset.from_generator(
                parse_bam_file,
                gen_kwargs={"file_path": self.hparams.val_data_path},
                num_proc=max(1, num_proc),
            ).with_format("torch")

            test_dataset = HuggingFaceDataset.from_generator(
                parse_bam_file,
                gen_kwargs={"file_path": self.hparams.test_data_path},
                num_proc=max(1, num_proc),
            ).with_format("torch")

            if self.hparams.max_train_samples is not None:
                max_train_samples = min(self.hparams.max_train_samples, len(train_dataset))
                train_dataset = HuggingFaceDataset.from_dict(train_dataset[:max_train_samples]).with_format("torch")

            if self.hparams.max_val_samples is not None:
                max_val_samples = min(self.hparams.max_val_samples, len(val_dataset))
                val_dataset = HuggingFaceDataset.from_dict(val_dataset[:max_val_samples]).with_format("torch")

            if self.hparams.max_test_samples is not None:
                max_test_samples = min(self.hparams.max_test_samples, len(test_dataset))
                test_dataset = HuggingFaceDataset.from_dict(test_dataset[:max_test_samples]).with_format("torch")

            self.data_train = train_dataset.map(
                partial(
                    tokenize_and_align_labels_and_quals,
                    tokenizer=self.hparams.tokenizer,
                    max_length=self.hparams.tokenizer.max_len_single_sentence,
                ),
                num_proc=max(1, num_proc),  # type: ignore
            ).remove_columns([SEQ_FEATURE, QUAL_FEATURE, ID_FEATURE])

            self.data_val = val_dataset.map(
                partial(
                    tokenize_and_align_labels_and_quals,
                    tokenizer=self.hparams.tokenizer,
                    max_length=self.hparams.tokenizer.max_len_single_sentence,
                ),
                num_proc=max(1, num_proc),  # type: ignore
            ).remove_columns([SEQ_FEATURE, QUAL_FEATURE, ID_FEATURE])

            self.data_test = test_dataset.map(
                partial(
                    tokenize_and_align_labels_and_quals,
                    tokenizer=self.hparams.tokenizer,
                    max_length=self.hparams.tokenizer.max_len_single_sentence,
                ),
                num_proc=max(1, num_proc),  # type: ignore
            ).remove_columns([SEQ_FEATURE, QUAL_FEATURE, ID_FEATURE])

            del train_dataset, val_dataset, test_dataset

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.data_collator.torch_call,
            shuffle=False,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.data_collator.torch_call,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.data_collator.torch_call,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        """Create and return the predict dataloader.

        :return: The predict dataloader.
        """
        return DataLoader(
            dataset=self.data_predict,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.data_collator.torch_call,
            shuffle=False,
        )

    def teardown(self, stage: str | None = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,.

        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """

    def state_dict(self) -> dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule.

        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
