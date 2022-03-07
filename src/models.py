import itertools
import nltk.data
import torch
import pytorch_lightning as pl

from typing import List, Iterable, Dict, Union, Tuple
from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_cosine_schedule_with_warmup,
)


class Scorer(ABC):
    def __init__(self):
        self.sentence_tokenizer = nltk.data.load("tokenizers/punkt/russian.pickle")

    @abstractmethod
    def score_sentences(self, sentences: List[str]) -> List[float]:
        raise NotImplementedError()

    @abstractmethod
    def fit(
        self,
        X_train: Iterable[str],
        y_train: Iterable[int],
        X_val: Union[Iterable[str], None] = None,
        y_val: Union[Iterable[int], None] = None,
        *args,
        **kwargs,
    ) -> None:
        raise NotImplementedError()

    @staticmethod
    def _delete_duplicates(
        sentences: Iterable[str], labels: Union[Iterable[int], None]
    ) -> Tuple[List[str], Union[List[int], None]]:
        if labels is not None:
            sentences, labels = zip(*set(zip(sentences, labels)))
            return sentences, labels
        else:
            return list(set(sentences)), None

    def score_news(self, news: Iterable[str]) -> List[float]:
        res = []
        for news_ in tqdm(news, desc="Scoring news..."):
            if isinstance(news_, float):
                res.append(0)
                continue

            sentences = self.sentence_tokenizer.tokenize(news_)
            news_prob = max(self.score_sentences(sentences))
            res.append(news_prob)

        return res

    def _preprocess(
        self, sentences: Iterable[str], labels: Union[Iterable[int], None]
    ) -> Tuple[List[str], Union[List[int], None]]:
        return self._delete_duplicates(sentences, labels)


class TfidfScorer(Scorer):
    def __init__(self, seed: int = 42):
        super().__init__()
        self.tf_idf = TfidfVectorizer()
        self.clf = LogisticRegression(
            penalty="l2",
            C=10,
            tol=1e-4,
            solver="lbfgs",
            max_iter=100,
            random_state=seed,
        )

    def fit(
        self,
        X_train: Iterable[str],
        y_train: Iterable[int],
        X_val: Union[Iterable[str], None] = None,
        y_val: Union[Iterable[int], None] = None,
        *args,
        **kwargs,
    ) -> None:
        X_train, y_train = self._preprocess(X_train, y_train)

        X_train = self.tf_idf.fit_transform(X_train)
        self.clf.fit(X_train, y_train)

    def score_sentences(self, sentences: List[str]) -> List[float]:
        sentences_transformed = self.tf_idf.transform(sentences)
        return self.clf.predict(sentences_transformed)


class BertScorer(Scorer):
    def __init__(
        self,
        lr: float = 2e-5,
        w_decay: float = 1e-5,
        warmup_steps: Union[int, float] = 0.1,
        scheduler: str = "cosine",
        seq_len: int = 128,
    ):
        super().__init__()
        self.model = BertClassifier(lr=lr, w_decay=w_decay, warmup_steps=warmup_steps, scheduler=scheduler)
        self.tokenizer = BertTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")

        self.seq_len = seq_len

    def _preprocess(self, sentences: Iterable[str], labels: Iterable[int] = None) -> TensorDataset:
        sentences, labels = super()._preprocess(sentences, labels)
        sentences_tokenized = self.tokenizer(
            sentences, truncation=True, padding=True, max_length=self.seq_len, return_tensors="pt"
        )

        if labels is not None:
            return TensorDataset(*sentences_tokenized.values(), torch.tensor(labels))
        else:
            return TensorDataset(*sentences_tokenized.values())

    def fit(
        self,
        X_train: Iterable[str],
        y_train: Iterable[int],
        X_val: Union[Iterable[str], None] = None,
        y_val: Union[Iterable[int], None] = None,
        n_epochs: int = 2,
        batch_size: int = 32,
        val_interval: float = 0.5,
        use_gpu: bool = True,
        seed: int = 42,
    ) -> None:
        pl.seed_everything(seed)

        train_ds = self._preprocess(X_train, y_train)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        if X_val is not None and y_val is not None:
            val_ds = self._preprocess(X_train, y_train)
            val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        else:
            val_dl = None

        trainer = pl.Trainer(
            val_check_interval=val_interval,
            max_epochs=n_epochs,
            gpus=int(use_gpu),
            progress_bar_refresh_rate=1,
        )

        trainer.fit(self.model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    @torch.no_grad()
    def score_sentences(self, sentences: List[str]) -> List[float]:
        ds = self._preprocess(sentences)
        dl = DataLoader(ds, batch_size=64, shuffle=False)

        scores = []
        for batch in dl:
            batch = [t.to(self.model.device) for t in batch]
            logits = self.model(*batch).logits

            scores.append(logits.softmax(-1).amax(-1).cpu().numpy().tolist())

        return list(itertools.chain(*scores))


class BertClassifier(pl.LightningModule):
    def __init__(
        self,
        lr: float = 2e-5,
        w_decay: float = 0.01,
        warmup_steps: Union[int, float] = 0.1,
        scheduler: str = "cosine",
    ):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained("DeepPavlov/rubert-base-cased")

        self.lr = lr
        self.w_decay = w_decay
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs

    @classmethod
    @torch.no_grad()
    def _compute_pr(cls, logits, target):
        preds = logits.argmax(-1)

        TP = ((preds == 1) & (target == 1)).count_nonzero()
        TN = ((preds == 0) & (target == 0)).count_nonzero()
        FP = ((preds == 1) & (target == 0)).count_nonzero()
        FN = ((preds == 0) & (target == 1)).count_nonzero()

        return {"TP": TP, "TN": TN, "FP": FP, "FN": FN}

    @torch.no_grad()
    def forward(self, input_ids, token_type_ids, attention_mask):
        self.model.eval()
        return self.model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

    def training_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, labels = batch
        res = self.model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)

        self.log("train_batch_loss", res.loss.item(), on_step=True, prog_bar=True, logger=True)

        return res.loss

    def validation_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, labels = batch
        res = self.model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)

        # calculating metrics
        metrics = self._compute_pr(res.logits, labels)

        return {"loss": res.loss.item(), **metrics}

    def validation_epoch_end(self, validation_step_outputs):
        val_epoch_loss = sum(map(lambda x: x["loss"], validation_step_outputs)) / len(validation_step_outputs)

        TP = sum(map(lambda x: x["TP"], validation_step_outputs))
        TN = sum(map(lambda x: x["TN"], validation_step_outputs))
        FP = sum(map(lambda x: x["FP"], validation_step_outputs))
        FN = sum(map(lambda x: x["FN"], validation_step_outputs))

        val_acc = (TP + TN) / (TP + TN + FN + FP)
        val_recall = TP / (TP + FN)
        val_precision = TP / (TP + FP)
        val_f1 = 2 * val_precision * val_recall / (val_precision + val_recall)

        self.log("val_loss", val_epoch_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", val_acc, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_recall", val_recall, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_precision", val_precision, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_f1", val_f1, on_epoch=True, prog_bar=True, logger=True)

        return {
            "val_loss": val_epoch_loss,
            "val_acc": val_acc,
            "val_recall": val_recall,
            "val_precision": val_precision,
            "val_f1": val_f1,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.w_decay)

        if isinstance(self.warmup_steps, float):
            warmup_steps = self.num_training_steps * self.warmup_steps
        else:
            warmup_steps = self.warmup_steps

        if self.scheduler == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=self.num_training_steps,
            )
            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

            return [optimizer], [scheduler]
        else:
            return optimizer
