from pathlib import Path
from typing import Any

import numpy as np
import pytest
from mlp.data_loader.actors import DataLoader
from mlp.dataset.prebuilts import EquationDataset, MnistDataset
from mlp.sampler.prebuilts import BatchSampler, RandomSampler
from mlp.trainer.prebuilts import (
    SecondOrderSupervisedLearningTrainer,
    SupervisedLearningTrainer,
)
from supervised_learning import (
    CrossEntropy,
    Linear,
    MeanSquaredError,
    Newton,
    ReLU,
    Sequential,
    StochasticGradientDescent,
)


@pytest.mark.public
def test_train_multi_layer_model_on_mnist_with_newton(
    multi_layer_model_for_mnist: Sequential,
    mnist_train_data_loader: DataLoader[Any, tuple[np.ndarray, np.ndarray]],
):
    trainer = SecondOrderSupervisedLearningTrainer(
        initial_model=multi_layer_model_for_mnist,
        initial_optimizer=Newton(epsilon=1.0),
        loss=CrossEntropy(),
    )

    _, losses = trainer.train(training_data=mnist_train_data_loader, epochs=1)

    assert len(losses) > 0, "No losses were recorded during training."
    assert np.std(losses) > 0, "Losses should vary during training."
    assert losses[-1] < losses[0], "Final loss should be less than initial loss."
    assert losses[-1] < 0.05, "Final loss is unexpectedly high."


@pytest.mark.public
def test_train_multi_layer_model_on_mnist_with_stochastic_gradient_descent(
    multi_layer_model_for_mnist: Sequential,
    mnist_train_data_loader: DataLoader[Any, tuple[np.ndarray, np.ndarray]],
):
    trainer = SupervisedLearningTrainer(
        initial_model=multi_layer_model_for_mnist,
        initial_optimizer=StochasticGradientDescent(learning_rate=0.01),
        loss=CrossEntropy(),
    )

    _, losses = trainer.train(training_data=mnist_train_data_loader, epochs=1)

    assert len(losses) > 0, "No losses were recorded during training."
    assert np.std(losses) > 0, "Losses should vary during training."
    assert losses[-1] < losses[0], "Final loss should be less than initial loss."
    assert losses[-1] < 0.5, "Final loss is unexpectedly high."


@pytest.mark.public
def test_train_single_layer_model_on_simple_dataset_with_newton(
    single_layer_model: Linear,
    simple_train_data_loader: DataLoader[Any, tuple[np.ndarray, np.ndarray]],
):
    loss = MeanSquaredError()
    optimizer = Newton()

    trainer = SecondOrderSupervisedLearningTrainer(
        initial_model=single_layer_model,
        initial_optimizer=optimizer,
        loss=loss,
    )

    _, losses = trainer.train(training_data=simple_train_data_loader, epochs=1)

    assert len(losses) > 0, "No losses were recorded during training."
    assert np.std(losses) > 0, "Losses should vary during training."
    assert losses[-1] < losses[0], "Final loss should be less than initial loss."
    assert losses[-1] < 0.001, "Final loss is unexpectedly high."


@pytest.mark.public
def test_train_single_layer_model_on_simple_dataset_with_stochastic_gradient_descent(
    single_layer_model: Linear,
    simple_train_data_loader: DataLoader[Any, tuple[np.ndarray, np.ndarray]],
):
    loss = MeanSquaredError()
    optimizer = StochasticGradientDescent(learning_rate=0.01)

    trainer = SupervisedLearningTrainer(
        initial_model=single_layer_model,
        initial_optimizer=optimizer,
        loss=loss,
    )

    _, losses = trainer.train(training_data=simple_train_data_loader, epochs=1)

    assert len(losses) > 0, "No losses were recorded during training."
    assert np.std(losses) > 0, "Losses should vary during training."
    assert losses[-1] < losses[0], "Final loss should be less than initial loss."
    assert losses[-1] < 1.0, "Final loss is unexpectedly high."


@pytest.fixture
def multi_layer_model_for_mnist() -> Sequential:
    return Sequential(
        modules=[
            Linear.create(
                input_size=28 * 28,
                output_size=128,
                random_generator=np.random.default_rng(42),
            ),
            ReLU(),
            Linear.create(
                input_size=128,
                output_size=128,
                random_generator=np.random.default_rng(42),
            ),
            ReLU(),
            Linear.create(
                input_size=128,
                output_size=10,
                random_generator=np.random.default_rng(42),
            ),
        ]
    )


@pytest.fixture
def single_layer_model() -> Linear:
    return Linear.create(
        input_size=1, output_size=1, random_generator=np.random.default_rng(42)
    )


@pytest.fixture
def mnist_train_data_loader(mnist_train_dataset: MnistDataset):
    return DataLoader(
        dataset=mnist_train_dataset,
        collate=MnistDataset.collate,
        sampler=BatchSampler(
            sampler=RandomSampler(
                data_source=mnist_train_dataset,
                random_generator=np.random.default_rng(42),
            ),
            batch_size=64,
            drop_last=True,
        ),
    )


@pytest.fixture
def mnist_train_dataset(cache_directory: Path):
    train_dataset, _ = MnistDataset.from_cache_directory_or_download(
        cache_directory=cache_directory
    )

    return train_dataset


@pytest.fixture
def simple_train_data_loader(simple_train_dataset: EquationDataset):
    return DataLoader(
        dataset=simple_train_dataset,
        collate=EquationDataset.collate,
        sampler=BatchSampler(
            sampler=RandomSampler(
                data_source=simple_train_dataset,
                random_generator=np.random.default_rng(42),
            ),
            batch_size=32,
            drop_last=True,
        ),
    )


@pytest.fixture
def simple_train_dataset() -> EquationDataset:
    def f(x: np.ndarray) -> np.ndarray:
        return 2 * x + 1

    return EquationDataset(
        f=f,
        length=1024,
        low=-16.0,
        high=16.0,
        random_generator=np.random.default_rng(42),
    )


@pytest.fixture
def cache_directory():
    directory = Path(__file__).parent.parent.parent.parent / ".cache"
    directory.mkdir(parents=True, exist_ok=True)
    return directory
