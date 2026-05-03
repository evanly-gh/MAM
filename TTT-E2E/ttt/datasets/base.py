"""Dataset-agnostic interface used by all personalization methods.

The point of this module is to make TTT, RAG, LoRA, and ICL
interchangeable across datasets. Every concrete dataset (LaMP-5,
LaMP-7, ELSA, anything else) reduces to the same shape:

    PersonalizedExample(user_id, profile, task_input, task_output)

Once a dataset can produce these, all downstream methods just work.

The two things every dataset MUST declare beyond producing examples:
  * task_type      -- "generation" vs "classification". Affects which
                      metrics make sense.
  * default_metric -- "bleu" / "rouge" / "meteor" / "accuracy" / "f1" /
                      "emotion_acc" / etc. The runner uses this when
                      the user doesn't pick one explicitly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, Literal


TaskType = Literal["generation", "classification"]


@dataclass
class PersonalizedExample:
    """A single (profile, query, answer) triple.

    Attributes:
        user_id: stable identifier for the persona / user. Two examples
            with the same user_id share the same profile semantics, so a
            method may cache per-user state across calls.
        profile: list of strings the method should adapt on / retrieve
            from / condition on. Each string is one "history item"
            (one past tweet, one past paper, one rewrite, ...). Order is
            irrelevant; methods that need an order define their own.
        task_input: the new query the model has to answer.
        task_output: gold answer (used only for scoring).
        metadata: free-form dict for dataset-specific extras (e.g.
            ``{"emotion": "joy", "style": "poetic"}`` for ELSA).
            Methods do not read this; the runner / scorers may.
    """

    user_id: str
    profile: list[str]
    task_input: str
    task_output: str
    metadata: dict = field(default_factory=dict)


class DatasetAdapter(ABC):
    """Subclass this once per dataset.

    Every method in TTT-E2E (TTT inner adapt, RAG, LoRA, ICL) consumes
    PersonalizedExample, so the only thing that changes when you swap
    datasets is which adapter you instantiate.
    """

    #: short identifier used by the registry (e.g. "elsa", "lamp-5").
    name: str

    #: "generation" if task_output is free text; "classification" if it
    #: is a label from a small set.
    task_type: TaskType

    #: metric to use when the runner wasn't given one explicitly.
    default_metric: str

    @abstractmethod
    def train_examples(self) -> Iterator[PersonalizedExample]:
        """Yield examples for meta-training / LoRA fine-tuning."""

    @abstractmethod
    def test_examples(self) -> Iterator[PersonalizedExample]:
        """Yield examples for evaluation. Each test example carries its
        own ``profile`` -- methods do not look across examples."""

    def info(self) -> dict:
        """Optional: metadata to print at run start (split sizes, etc)."""
        return {"name": self.name, "task_type": self.task_type, "metric": self.default_metric}
