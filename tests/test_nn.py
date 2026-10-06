import heyoka as hk
import numpy as np
import pytest
import torch

import eclipsenets as en

BODIES = ["Bennu", "Itokawa", "67P", "Eros"]


def heyoka_outputs(model, inputs, weights=None, pars=()):
    variables = hk.make_vars(*[f"u{i}" for i in range(inputs.shape[1])])
    compiled = hk.cfunc(en.heyoka_model(model, variables, weights), variables)
    return np.array([compiled(row, pars=pars)[0] for row in inputs])


@pytest.mark.parametrize("body", BODIES)
@pytest.mark.parametrize("kind", ["siren", "ffnn"])
def test_pretrained_models(body, kind):
    model = en.load_pretrained(body, kind)
    assert en.count_parameters(model) == 2369  # as in the paper
    assert not model.training

    # the heyoka expression gives the same values as the torch model
    inputs = np.random.default_rng(0).uniform(-0.6, 0.6, (20, 6))
    with torch.no_grad():
        expected = model.double()(torch.tensor(inputs)).numpy().ravel()
    assert heyoka_outputs(model, inputs) == pytest.approx(expected, abs=1e-12)


def test_intermediate_checkpoints():
    for epoch in [0, 7]:
        assert en.load_pretrained("67P", "ffnn", epoch=epoch).activation == "tanh"
    with pytest.raises(FileNotFoundError):
        en.pretrained_path("Bennu", "ffnn", epoch=0)


@pytest.mark.parametrize("model", [
    en.SIREN(3, 8, 1, 1),
    en.SIREN(3, 8, 2, 2, outermost_linear=False, first_omega_0=10., hidden_omega_0=20.),
    en.FFNN(3, [8, 5], 1, activation="relu"),
    en.FFNN(3, [8], 1, activation=torch.nn.Tanh()),
    en.FFNN(3, [8, 8], 1, activation="sigmoid"),
])
def test_heyoka_model_with_runtime_weights(model):
    model.double()
    inputs = np.random.default_rng(0).uniform(-1, 1, (10, 3))
    with torch.no_grad():
        expected = model(torch.tensor(inputs)).numpy()[:, 0]

    weights = en.weights_and_biases_heyoka(model)
    assert len(weights) == en.count_parameters(model)
    assert heyoka_outputs(model, inputs) == pytest.approx(expected, abs=1e-12)
    pars = [hk.par[i] for i in range(len(weights))]
    assert heyoka_outputs(model, inputs, weights=pars, pars=weights) == pytest.approx(expected, abs=1e-12)


def test_training_and_checkpoints(tmp_path):
    # a toy eclipse function: the signed distance from a circle
    rng = np.random.default_rng(0)
    inputs = rng.uniform(-1, 1, (4096, 6))
    data = np.column_stack([inputs, np.hypot(inputs[:, 4], inputs[:, 5]) - 0.5])

    for model_type, activation in [("siren", "relu"), ("ffnn", "tanh")]:
        config = en.TrainingConfig(model_type=model_type, activation=activation, epochs=3, batch_size=64,
                                   seed=0, device="cpu", checkpoint_dir=str(tmp_path), save_frequency=2)
        model, history = en.train(data[:3072], data[3072:], config, progress=False)

        assert len(history["train_losses"]) == len(history["valid_losses"]) == 3
        assert len(history["batch_losses"]) == 3 * 48
        assert history["train_losses"][-1] < history["train_losses"][0]
        assert history["best_valid_loss"] == min(history["valid_losses"])
        assert en.evaluate_mse(model, data[3072:]) == pytest.approx(history["valid_losses"][-1])

        # the checkpoints carry what is needed to rebuild the model
        assert (tmp_path / f"{model_type}_history.pkl").exists()
        last = en.load_model(tmp_path / f"{model_type}_epoch_2.pt")
        best = en.load_model(tmp_path / f"{model_type}_best.pt")
        assert type(last) is type(best) is type(model)
        assert getattr(best, "activation", None) == (activation if model_type == "ffnn" else None)
        assert en.evaluate_mse(best, data[3072:]) == pytest.approx(history["best_valid_loss"])

    # the caller's objects are left alone
    assert config.input_dim == 6
    model.train()
    en.evaluate_mse(model, data[3072:])
    assert model.training


def test_load_model_keeps_the_precision(tmp_path):
    model = en.FFNN(6, [8], 1, activation="tanh").double()
    torch.save(model.state_dict(), tmp_path / "model.pt")
    loaded = en.load_model(tmp_path / "model.pt", activation="tanh")
    inputs = torch.rand(5, 6, dtype=torch.float64)
    assert torch.equal(loaded(inputs), model(inputs))


def test_load_dataset_reports_missing_lfs_files(tmp_path):
    data = np.arange(14.).reshape(2, 7)
    en.save_dataset(data, tmp_path / "data.pk")
    assert np.array_equal(en.load_dataset(tmp_path / "data.pk"), data)

    pointer = tmp_path / "datasets" / "pointer.pk"
    pointer.parent.mkdir()
    oid = "ab" + "0" * 62
    pointer.write_text(f"version https://git-lfs.github.com/spec/v1\noid sha256:{oid}\nsize 10\n")
    with pytest.raises(FileNotFoundError, match="git lfs pull"):
        en.load_dataset(pointer)

    # the file is found if git lfs has fetched it without checking it out
    cache = tmp_path / ".git" / "lfs" / "objects" / "ab" / "00"
    cache.mkdir(parents=True)
    en.save_dataset(data, cache / oid)
    assert np.array_equal(en.load_dataset(pointer), data)
