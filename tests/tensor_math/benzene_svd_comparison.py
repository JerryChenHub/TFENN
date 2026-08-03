"""Compare matched benzene force models including group convolution."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from torch import Tensor, nn

from TFENN.data import BenzenePairDataset
from TFENN.tensor_math import PoseEncoder, compile_anchors, compile_intertwiners


DTYPE = torch.float64
DATA_FILE = "Benzene_10000_5.0_10.0_3.0_gamma1.csv"
SPLIT_SEED = 20260802
MODEL_SEED = 20260803


def benzene_generators() -> Tensor:
    """Return generators for the proper rotational D6 action."""
    cosine = math.cos(math.pi / 3.0)
    sine = math.sin(math.pi / 3.0)
    return torch.tensor(
        (
            (
                (cosine, -sine, 0.0),
                (sine, cosine, 0.0),
                (0.0, 0.0, 1.0),
            ),
            (
                (1.0, 0.0, 0.0),
                (0.0, -1.0, 0.0),
                (0.0, 0.0, -1.0),
            ),
        ),
        dtype=DTYPE,
    )


def d6_rotations() -> Tensor:
    """Return all twelve proper rotations in deterministic order."""
    rotation, flip = benzene_generators()
    identity = torch.eye(3, dtype=DTYPE)
    powers = [identity]
    for _ in range(5):
        powers.append(powers[-1] @ rotation)
    return torch.stack((*powers, *(value @ flip for value in powers)))


def relative_product_table(rotations: Tensor) -> Tensor:
    """Return the index of each inverse first times second product."""
    relative = rotations.mT[:, None] @ rotations[None]
    distances = torch.linalg.matrix_norm(
        relative[:, :, None] - rotations[None, None],
        dim=(-2, -1),
    )
    minimum, indices = distances.min(dim=-1)
    if float(minimum.max()) > 2.0e-12:
        raise RuntimeError("D6 multiplication table is not closed")
    return indices


def regular_probes(
    rotations: Tensor,
    representation_b: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return fixed complete probes with identity orbit frame operators."""
    probe_a = (
        torch.tensor(
            (math.sqrt(2.0 / 3.0), 0.0, math.sqrt(1.0 / 3.0)),
            dtype=DTYPE,
        )
        / 2.0
    )
    orbit_a = torch.einsum("hij,j->hi", rotations, probe_a)
    identity_a = torch.eye(3, dtype=DTYPE)
    if not torch.allclose(orbit_a.mT @ orbit_a, identity_a, atol=3e-12, rtol=3e-12):
        raise RuntimeError("A probe orbit is not a normalized frame")

    generator = torch.Generator().manual_seed(4)
    raw = torch.randn(3, representation_b.shape[-1], dtype=DTYPE, generator=generator)
    raw = raw / torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
    raw_orbit = torch.einsum("hij,pj->hpi", representation_b, raw)
    frame = torch.einsum("hpi,hpj->ij", raw_orbit, raw_orbit)
    eigenvalues, eigenvectors = torch.linalg.eigh(frame)
    if float(eigenvalues.min()) <= 1.0e-12:
        raise RuntimeError("B probes do not span the pose representation")
    inverse_root = eigenvectors @ torch.diag(eigenvalues.rsqrt()) @ eigenvectors.mT
    probes_b = raw @ inverse_root
    orbit_b = torch.einsum("hij,pj->hpi", representation_b, probes_b)
    measurement = orbit_b.reshape(-1, representation_b.shape[-1])
    identity_b = torch.eye(representation_b.shape[-1], dtype=DTYPE)
    if int(torch.linalg.matrix_rank(measurement)) != representation_b.shape[-1]:
        raise RuntimeError("B probe measurement is not injective")
    if not torch.allclose(
        measurement.mT @ measurement,
        identity_b,
        atol=5e-12,
        rtol=5e-12,
    ):
        raise RuntimeError("B probe orbit is not a normalized frame")
    return probe_a, probes_b


class TypedIntertwinerLinear(nn.Module):
    """Mix channels through one fixed intertwiner basis."""

    def __init__(
        self,
        basis: Tensor,
        input_channels: int,
        output_channels: int,
        generator: torch.Generator,
        *,
        identity: bool = False,
        scale: float = 0.2,
    ) -> None:
        """Initialize one typed linear map without bias."""
        super().__init__()
        self.register_buffer("basis", basis.clone())
        standard_deviation = scale / math.sqrt(input_channels * basis.shape[0])
        self.weight = nn.Parameter(
            standard_deviation
            * torch.randn(
                output_channels,
                input_channels,
                basis.shape[0],
                dtype=basis.dtype,
                generator=generator,
            )
        )
        if identity:
            if input_channels != output_channels:
                raise ValueError("identity initialization requires equal channels")
            coefficients = torch.einsum("kii->k", basis)
            with torch.no_grad():
                self.weight.zero_()
                for channel in range(input_channels):
                    self.weight[channel, channel].copy_(coefficients)

    def forward(self, value: Tensor) -> Tensor:
        """Apply the learned basis mixture."""
        return torch.einsum(
            "ock,kdi,bci->bod",
            self.weight,
            self.basis,
            value,
        )


class SVDBlockActivation(nn.Module):
    """Apply tanh to singular values within representation blocks."""

    def __init__(self, block_sizes: tuple[int, ...]) -> None:
        """Store a complete partition of the representation axis."""
        super().__init__()
        if not block_sizes or any(size < 1 for size in block_sizes):
            raise ValueError("block sizes must be positive")
        self.block_sizes = block_sizes

    def forward(self, value: Tensor) -> Tensor:
        """Apply a zero preserving rectangular spectral map per block."""
        if value.ndim != 3 or value.shape[-1] != sum(self.block_sizes):
            raise ValueError("value does not match the SVD block layout")
        blocks = value.split(self.block_sizes, dim=-1)
        activated = []
        for block in blocks:
            left, singular_values, right = torch.linalg.svd(
                block,
                full_matrices=False,
            )
            activated.append((left * torch.tanh(singular_values).unsqueeze(-2)) @ right)
        return torch.cat(activated, dim=-1)


class FiniteGroupConvolution(nn.Module):
    """Apply one full right convolution on a finite regular field."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        relative_index: Tensor,
        generator: torch.Generator,
    ) -> None:
        """Initialize one complete group kernel without bias."""
        super().__init__()
        group_order = relative_index.shape[0]
        if relative_index.shape != (group_order, group_order):
            raise ValueError("relative index must be a square table")
        self.input_channels = input_channels
        self.register_buffer("relative_index", relative_index.clone())
        self.weight = nn.Parameter(
            torch.randn(
                output_channels,
                input_channels,
                group_order,
                dtype=DTYPE,
                generator=generator,
            )
            / math.sqrt(input_channels * group_order)
        )

    def forward(self, value: Tensor) -> Tensor:
        """Convolve a batch by the complete relative group table."""
        if (
            value.ndim != 3
            or value.shape[1] != self.relative_index.shape[0]
            or value.shape[2] != self.input_channels
        ):
            raise ValueError("value does not match the regular field layout")
        expanded_kernel = self.weight[:, :, self.relative_index]
        return torch.einsum("oiht,nti->nho", expanded_kernel, value)


class D6GroupConvMLP(nn.Module):
    """Predict force with complete D6 convolutions and pointwise SiLU."""

    def __init__(
        self,
        encoder: PoseEncoder,
        rotations: Tensor,
        representation_b: Tensor,
        *,
        seed: int,
    ) -> None:
        """Build an injective regular lift and four group convolutions."""
        super().__init__()
        if rotations.shape != (12, 3, 3):
            raise ValueError("D6 rotations must have shape twelve by three by three")
        if representation_b.shape != (12, 18, 18):
            raise ValueError("D6 B representation must have shape twelve by eighteen")
        probe_a, probes_b = regular_probes(rotations, representation_b)
        self.pose_encoder = encoder
        self.register_buffer("rotations", rotations.clone())
        self.register_buffer("representation_b", representation_b.clone())
        self.register_buffer("probe_a", probe_a)
        self.register_buffer("probes_b", probes_b)
        self.register_buffer(
            "orbit_a",
            torch.einsum("hij,j->hi", rotations, probe_a),
        )
        relative_index = relative_product_table(rotations)
        generator = torch.Generator().manual_seed(seed)
        widths = ((4, 3), (3, 3), (3, 2), (2, 1))
        self.convolutions = nn.ModuleList(
            FiniteGroupConvolution(
                input_channels,
                output_channels,
                relative_index,
                generator,
            )
            for input_channels, output_channels in widths
        )

    def force_from_pose(self, displacement: Tensor, pose: Tensor) -> Tensor:
        """Evaluate force from displacement and sender invariant pose."""
        if displacement.ndim != 2 or displacement.shape[-1] != 3:
            raise ValueError("displacement must have shape batch by three")
        if pose.shape != (displacement.shape[0], 18):
            raise ValueError("pose must have shape batch by eighteen")
        field_a = torch.einsum(
            "ni,hij,j->nh",
            displacement,
            self.rotations,
            self.probe_a,
        ).unsqueeze(-1)
        field_b = torch.einsum(
            "ni,hij,pj->nhp",
            pose,
            self.representation_b,
            self.probes_b,
        )
        value = torch.cat((field_a, field_b), dim=-1)
        for convolution in self.convolutions[:-1]:
            value = torch.nn.functional.silu(convolution(value))
        value = self.convolutions[-1](value)
        return torch.einsum("nh,hi->ni", value[..., 0], self.orbit_a)

    def forward(self, displacement: Tensor, rotation: Tensor) -> Tensor:
        """Encode relative pose and evaluate normalized force."""
        return self.force_from_pose(displacement, self.pose_encoder(rotation))


class BenzeneForceModel(nn.Module):
    """Use matched A and B paths with optional SVD block activations."""

    def __init__(
        self,
        encoder: PoseEncoder,
        lift_aa: Tensor,
        lift_ba: Tensor,
        lift_bb: Tensor,
        *,
        channels: int,
        seed: int,
        use_svd: bool,
    ) -> None:
        """Build the requested four A stages and three B stages."""
        super().__init__()
        self.pose_encoder = encoder
        self.use_svd = use_svd
        generator = torch.Generator().manual_seed(seed)

        self.a1 = TypedIntertwinerLinear(lift_aa, 1, channels, generator, scale=0.4)
        self.a2 = TypedIntertwinerLinear(
            lift_aa, channels, channels, generator, identity=True
        )
        self.a3 = TypedIntertwinerLinear(
            lift_aa, channels, channels, generator, identity=True
        )
        self.b1 = TypedIntertwinerLinear(lift_bb, 1, channels, generator, scale=0.4)
        self.b2 = TypedIntertwinerLinear(
            lift_bb, channels, channels, generator, identity=True
        )
        self.b3 = TypedIntertwinerLinear(
            lift_bb, channels, channels, generator, identity=True
        )
        self.a4_from_a = TypedIntertwinerLinear(
            lift_aa, channels, channels, generator, identity=True
        )
        self.a4_from_b = TypedIntertwinerLinear(
            lift_ba, channels, channels, generator, scale=0.12
        )
        self.force_head = TypedIntertwinerLinear(
            lift_aa, channels, 1, generator, scale=0.4
        )
        self.activation_a: nn.Module
        self.activation_b: nn.Module
        if use_svd:
            self.activation_a = SVDBlockActivation((3,))
            self.activation_b = SVDBlockActivation((5, 13))
        else:
            self.activation_a = nn.Identity()
            self.activation_b = nn.Identity()

    def force_from_pose(self, displacement: Tensor, pose: Tensor) -> Tensor:
        """Evaluate normalized receiver frame force from cached pose."""
        value_a = self.activation_a(self.a1(displacement[:, None]))
        value_a = self.activation_a(self.a2(value_a))
        value_a = self.activation_a(self.a3(value_a))

        value_b = self.activation_b(self.b1(pose[:, None]))
        value_b = self.activation_b(self.b2(value_b))
        value_b = self.activation_b(self.b3(value_b))

        value_a4 = self.a4_from_a(value_a) + self.a4_from_b(value_b)
        value_a4 = self.activation_a(value_a4)
        return self.force_head(value_a4)[:, 0]

    def forward(self, displacement: Tensor, rotation: Tensor) -> Tensor:
        """Encode relative pose and evaluate normalized force."""
        return self.force_from_pose(
            displacement,
            self.pose_encoder(rotation),
        )


def compile_models(channels: int) -> dict[str, nn.Module]:
    """Compile shared constants and three independent matched models."""
    generators = benzene_generators()
    encoder = PoseEncoder(compile_anchors(generators, ranks=(2, 6)))
    representation_b = encoder.representation(generators)
    lift_aa = compile_intertwiners(generators, generators)
    lift_ba = compile_intertwiners(representation_b, generators)
    lift_bb = compile_intertwiners(representation_b, representation_b)
    if lift_aa.shape != (2, 3, 3):
        raise RuntimeError("unexpected AA dimension")
    if lift_ba.shape != (4, 3, 18):
        raise RuntimeError("unexpected BA dimension")
    if lift_bb.shape != (30, 18, 18):
        raise RuntimeError("unexpected BB dimension")
    rotations = d6_rotations()
    models: dict[str, nn.Module] = {
        "linear": BenzeneForceModel(
            copy.deepcopy(encoder),
            lift_aa,
            lift_ba,
            lift_bb,
            channels=channels,
            seed=MODEL_SEED,
            use_svd=False,
        ),
        "svd": BenzeneForceModel(
            copy.deepcopy(encoder),
            lift_aa,
            lift_ba,
            lift_bb,
            channels=channels,
            seed=MODEL_SEED,
            use_svd=True,
        ),
        "group_conv": D6GroupConvMLP(
            copy.deepcopy(encoder),
            rotations,
            encoder.representation(rotations),
            seed=MODEL_SEED,
        ),
    }
    parameter_counts = {
        name: sum(parameter.numel() for parameter in model.parameters())
        for name, model in models.items()
    }
    if set(parameter_counts.values()) != {348}:
        raise RuntimeError(f"model parameters are not matched: {parameter_counts}")
    return models


def load_split(
    repository_root: Path,
    training_count: int,
    validation_count: int,
) -> tuple[BenzenePairDataset, Tensor, Tensor, Tensor]:
    """Create one fixed label independent train and validation split."""
    dataset = BenzenePairDataset(
        repository_root / "data" / "benzene_pair" / DATA_FILE,
        dtype=DTYPE,
    )
    permutation = torch.randperm(
        len(dataset),
        generator=torch.Generator().manual_seed(SPLIT_SEED),
    )
    training = permutation[:training_count]
    training_evaluation = training[-validation_count:]
    validation = permutation[training_count : training_count + validation_count]
    if set(training.tolist()) & set(validation.tolist()):
        raise RuntimeError("training and validation indices overlap")
    return dataset, training, training_evaluation, validation


def train_model(
    name: str,
    model: BenzeneForceModel | D6GroupConvMLP,
    displacement: Tensor,
    pose: Tensor,
    target: Tensor,
    validation_displacement: Tensor,
    validation_pose: Tensor,
    validation_target: Tensor,
    *,
    steps: int,
    learning_rate: float,
    force_scale: Tensor,
) -> dict[str, Any]:
    """Train one model and retain every full batch loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    normalized_history: list[float] = []
    physical_history: list[float] = []
    normalized_validation_history: list[float] = []
    physical_validation_history: list[float] = []
    maximum_gradient = 0.0
    minimum_initial_gradient = math.inf
    started = time.perf_counter()

    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = model.force_from_pose(displacement, pose)
        loss = torch.nn.functional.mse_loss(prediction, target)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"{name} loss became nonfinite at step {step}")
        normalized_history.append(float(loss.detach()))
        physical_history.append(float(loss.detach() * force_scale.square()))
        with torch.no_grad():
            validation_prediction = model.force_from_pose(
                validation_displacement,
                validation_pose,
            )
            validation_loss = torch.nn.functional.mse_loss(
                validation_prediction,
                validation_target,
            )
        if not bool(torch.isfinite(validation_loss)):
            raise RuntimeError(
                f"{name} validation loss became nonfinite at step {step}"
            )
        normalized_validation_history.append(float(validation_loss))
        physical_validation_history.append(
            float(validation_loss * force_scale.square())
        )
        if step % 250 == 0:
            print(
                f"{name} step {step} "
                f"normalized_mse {normalized_history[-1]:.10g} "
                f"force_mse {physical_history[-1]:.10g} "
                f"validation_force_mse {physical_validation_history[-1]:.10g}"
            )
        if step == steps:
            break

        loss.backward()
        named_parameters = dict(model.named_parameters())
        gradients = {
            parameter_name: parameter.grad
            for parameter_name, parameter in named_parameters.items()
        }
        missing = [
            parameter_name
            for parameter_name, gradient in gradients.items()
            if gradient is None
        ]
        if missing:
            raise RuntimeError(f"{name} missing gradients at step {step}: {missing}")
        for parameter_name, gradient in gradients.items():
            if gradient is None or not bool(torch.isfinite(gradient).all()):
                raise RuntimeError(
                    f"{name} gradient became nonfinite at step {step}: {parameter_name}"
                )
            maximum_gradient = max(
                maximum_gradient,
                float(gradient.detach().abs().max()),
            )
        if step == 0:
            minimum_initial_gradient = min(
                float(torch.linalg.vector_norm(gradient.detach()))
                for gradient in gradients.values()
                if gradient is not None
            )
            if minimum_initial_gradient <= 0.0:
                raise RuntimeError(f"{name} has a zero initial parameter gradient")
        optimizer.step()
        if not all(bool(torch.isfinite(value).all()) for value in model.parameters()):
            raise RuntimeError(f"{name} parameters became nonfinite at step {step}")

    return {
        "normalized_mse": normalized_history,
        "force_mse": physical_history,
        "normalized_validation_mse": normalized_validation_history,
        "validation_force_mse": physical_validation_history,
        "maximum_gradient": maximum_gradient,
        "minimum_initial_gradient": minimum_initial_gradient,
        "seconds": time.perf_counter() - started,
    }


def evaluate(
    model: BenzeneForceModel | D6GroupConvMLP,
    dataset: BenzenePairDataset,
    indices: Tensor,
    displacement_scale: Tensor,
    force_scale: Tensor,
) -> dict[str, Any]:
    """Evaluate physical force metrics and retain sample predictions."""
    displacement = dataset.displacement[indices] / displacement_scale
    rotation = dataset.relative_rotation[indices]
    target = dataset.target[indices, 0]
    with torch.no_grad():
        prediction = model(displacement, rotation) * force_scale
    error = prediction - target
    mse = error.square().mean()
    rmse = torch.sqrt(mse)
    target_rms = torch.sqrt(target.square().mean())
    samples = []
    for index, expected, actual, difference in zip(
        indices.tolist(),
        target.tolist(),
        prediction.tolist(),
        error.tolist(),
    ):
        samples.append(
            {
                "index": index,
                "target": expected,
                "prediction": actual,
                "error_norm": math.sqrt(sum(value * value for value in difference)),
            }
        )
    return {
        "count": len(indices),
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(error.abs().mean()),
        "maximum_absolute_error": float(error.abs().max()),
        "target_rms": float(target_rms),
        "relative_rmse": float(rmse / target_rms),
        "zero_prediction_mse": float(target.square().mean()),
        "samples": samples,
    }


def gradient_probe(
    model: BenzeneForceModel | D6GroupConvMLP,
    displacement: Tensor,
    rotation: Tensor,
    target: Tensor,
) -> dict[str, float]:
    """Check the complete pose path after training."""
    model.zero_grad(set_to_none=True)
    probe_x = displacement.clone().requires_grad_(True)
    probe_r = rotation.clone().requires_grad_(True)
    loss = torch.nn.functional.mse_loss(model(probe_x, probe_r), target)
    loss.backward()
    if probe_x.grad is None or not bool(torch.isfinite(probe_x.grad).all()):
        raise RuntimeError("displacement probe gradient is not finite")
    if probe_r.grad is None or not bool(torch.isfinite(probe_r.grad).all()):
        raise RuntimeError("rotation probe gradient is not finite")
    parameter_gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    if not parameter_gradients or not all(
        bool(torch.isfinite(value).all()) for value in parameter_gradients
    ):
        raise RuntimeError("parameter probe gradients are not finite")
    return {
        "loss": float(loss.detach()),
        "displacement_gradient_norm": float(torch.linalg.vector_norm(probe_x.grad)),
        "rotation_gradient_norm": float(torch.linalg.vector_norm(probe_r.grad)),
        "maximum_parameter_gradient": max(
            float(value.detach().abs().max()) for value in parameter_gradients
        ),
    }


def symmetry_residual(
    model: BenzeneForceModel | D6GroupConvMLP,
    displacement: Tensor,
    rotation: Tensor,
) -> float:
    """Check every receiver and sender action in the complete group."""
    actions = d6_rotations()
    maximum = 0.0
    with torch.no_grad():
        reference = model(displacement, rotation)
        for receiver in actions:
            for sender in actions:
                transformed = model(
                    displacement @ receiver,
                    receiver.mT @ rotation @ sender,
                )
                expected = reference @ receiver
                maximum = max(
                    maximum,
                    float((transformed - expected).abs().max()),
                )
    return maximum


def plot_histories(histories: dict[str, dict[str, Any]], path: Path) -> None:
    """Plot physical training and validation losses on one logarithmic axis."""
    width, height = 1440, 900
    left, right, top, bottom = 120, 50, 70, 100
    plot_width = width - left - right
    plot_height = height - top - bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    series = {
        "LINEAR TRAIN": (histories["linear"]["force_mse"], "#2468b4"),
        "LINEAR VALIDATION": (
            histories["linear"]["validation_force_mse"],
            "#73a9d8",
        ),
        "SVD TRAIN": (histories["svd"]["force_mse"], "#d1492e"),
        "SVD VALIDATION": (
            histories["svd"]["validation_force_mse"],
            "#e79b57",
        ),
        "GROUP CONV TRAIN": (
            histories["group_conv"]["force_mse"],
            "#238b57",
        ),
        "GROUP CONV VALIDATION": (
            histories["group_conv"]["validation_force_mse"],
            "#82c89c",
        ),
    }
    all_losses = [
        max(float(value), torch.finfo(DTYPE).tiny)
        for losses, _color in series.values()
        for value in losses
    ]
    minimum_power = math.floor(math.log10(min(all_losses)))
    maximum_power = math.ceil(math.log10(max(all_losses)))
    if minimum_power == maximum_power:
        minimum_power -= 1

    def x_coordinate(step: int, count: int) -> float:
        return left + plot_width * step / max(count - 1, 1)

    def y_coordinate(value: float) -> float:
        logarithm = math.log10(max(value, torch.finfo(DTYPE).tiny))
        fraction = (maximum_power - logarithm) / (maximum_power - minimum_power)
        return top + plot_height * fraction

    for power in range(minimum_power, maximum_power + 1):
        y_value = y_coordinate(10.0**power)
        draw.line(
            (left, y_value, width - right, y_value),
            fill="#d8d8d8",
            width=1,
        )
        draw.text((15, y_value - 8), f"10^{power}", fill="black")
    draw.line((left, top, left, height - bottom), fill="black", width=2)
    draw.line(
        (left, height - bottom, width - right, height - bottom),
        fill="black",
        width=2,
    )

    for _name, (losses, color) in series.items():
        points = [
            (x_coordinate(step, len(losses)), y_coordinate(float(value)))
            for step, value in enumerate(losses)
        ]
        draw.line(points, fill=color, width=4)

    draw.text(
        (width // 2 - 160, 20),
        "Benzene train and validation force MSE",
        fill="black",
    )
    draw.text((width // 2 - 70, height - 45), "Optimization step", fill="black")
    draw.text((15, 25), "Log scale", fill="black")
    legend_x = width - right - 300
    for row, (name, (_losses, color)) in enumerate(series.items()):
        legend_y = top + 20 + 35 * row
        draw.line(
            (legend_x, legend_y, legend_x + 55, legend_y),
            fill=color,
            width=5,
        )
        draw.text((legend_x + 70, legend_y - 8), name, fill="black")
    image.save(path)


def plot_predictions(results: dict[str, Any], path: Path) -> None:
    """Plot final target and prediction components for both evaluations."""
    width, height = 1800, 1100
    left, right, top, bottom = 100, 35, 135, 70
    column_gap, row_gap = 45, 110
    panel_width = (width - left - right - 2 * column_gap) / 3.0
    panel_height = (height - top - bottom - row_gap) / 2.0
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    colors = {
        "target": "#111111",
        "linear": "#2468b4",
        "svd": "#d1492e",
        "group_conv": "#238b57",
    }
    draw.text(
        (width // 2 - 150, 20),
        "Final benzene force predictions",
        fill="black",
    )
    legend_x = width - right - 570
    for column, name in enumerate(("target", "linear", "svd", "group_conv")):
        x_value = legend_x + 140 * column
        draw.line((x_value, 60, x_value + 35, 60), fill=colors[name], width=5)
        draw.text((x_value + 42, 52), name.upper(), fill="black")

    all_values = []
    for subset in ("training_evaluation", "validation"):
        for model_name in ("linear", "svd", "group_conv"):
            for sample in results[model_name][subset]["samples"]:
                all_values.extend(sample["target"])
                all_values.extend(sample["prediction"])
    minimum = min(all_values)
    maximum = max(all_values)
    padding = max(0.05 * (maximum - minimum), 1.0e-3)
    minimum -= padding
    maximum += padding

    for row, subset in enumerate(("training_evaluation", "validation")):
        linear_result = results["linear"][subset]
        svd_result = results["svd"][subset]
        group_result = results["group_conv"][subset]
        linear_samples = linear_result["samples"]
        svd_samples = svd_result["samples"]
        group_samples = group_result["samples"]
        row_name = "Training" if row == 0 else "Validation"
        summary = (
            f"{row_name}  Linear MSE {linear_result['mse']:.5g}  "
            f"SVD MSE {svd_result['mse']:.5g}  "
            f"Group Conv MSE {group_result['mse']:.5g}  "
            f"Group Conv relative RMSE {group_result['relative_rmse']:.4g}"
        )
        panel_top = top + row * (panel_height + row_gap)
        draw.text((left, panel_top - 48), summary, fill="black")

        for component in range(3):
            panel_left = left + component * (panel_width + column_gap)
            panel_right = panel_left + panel_width
            panel_bottom = panel_top + panel_height

            def x_coordinate(index: int) -> float:
                return panel_left + panel_width * index / max(
                    len(linear_samples) - 1,
                    1,
                )

            def y_coordinate(value: float) -> float:
                return panel_bottom - panel_height * (value - minimum) / (
                    maximum - minimum
                )

            for tick in range(5):
                fraction = tick / 4.0
                value = maximum - fraction * (maximum - minimum)
                y_value = panel_top + fraction * panel_height
                draw.line(
                    (panel_left, y_value, panel_right, y_value),
                    fill="#dddddd",
                    width=1,
                )
                if component == 0:
                    draw.text(
                        (panel_left - 70, y_value - 8),
                        f"{value:.3f}",
                        fill="black",
                    )
            draw.line(
                (panel_left, panel_top, panel_left, panel_bottom),
                fill="black",
                width=2,
            )
            draw.line(
                (panel_left, panel_bottom, panel_right, panel_bottom),
                fill="black",
                width=2,
            )
            series = {
                "target": [sample["target"][component] for sample in linear_samples],
                "linear": [
                    sample["prediction"][component] for sample in linear_samples
                ],
                "svd": [sample["prediction"][component] for sample in svd_samples],
                "group_conv": [
                    sample["prediction"][component] for sample in group_samples
                ],
            }
            for name, values_to_plot in series.items():
                points = [
                    (x_coordinate(index), y_coordinate(float(value)))
                    for index, value in enumerate(values_to_plot)
                ]
                draw.line(points, fill=colors[name], width=3)
                for x_value, y_value in points:
                    if name == "target":
                        draw.ellipse(
                            (x_value - 4, y_value - 4, x_value + 4, y_value + 4),
                            fill=colors[name],
                        )
                    elif name == "linear":
                        draw.rectangle(
                            (x_value - 4, y_value - 4, x_value + 4, y_value + 4),
                            fill=colors[name],
                        )
                    elif name == "svd":
                        draw.polygon(
                            (
                                (x_value, y_value - 5),
                                (x_value - 5, y_value + 4),
                                (x_value + 5, y_value + 4),
                            ),
                            fill=colors[name],
                        )
                    else:
                        draw.polygon(
                            (
                                (x_value, y_value - 5),
                                (x_value - 5, y_value),
                                (x_value, y_value + 5),
                                (x_value + 5, y_value),
                            ),
                            fill=colors[name],
                        )
            draw.text(
                (panel_left + panel_width / 2 - 15, panel_top - 22),
                f"F{component + 1}",
                fill="black",
            )
            for sample_index in range(len(linear_samples)):
                draw.text(
                    (x_coordinate(sample_index) - 12, panel_bottom + 12),
                    str(linear_samples[sample_index]["index"]),
                    fill="black",
                )
    draw.text((width // 2 - 70, height - 28), "Dataset index", fill="black")
    image.save(path)


def run(config: argparse.Namespace) -> dict[str, Any]:
    """Run the fixed comparison and write its complete record."""
    repository_root = Path(__file__).resolve().parents[2]
    output_directory = config.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.threads)

    (
        dataset,
        training_indices,
        training_evaluation_indices,
        validation_indices,
    ) = load_split(
        repository_root,
        config.training_count,
        config.validation_count,
    )
    training_x = dataset.displacement[training_indices]
    training_r = dataset.relative_rotation[training_indices]
    training_y = dataset.target[training_indices, 0]
    displacement_scale = torch.sqrt(training_x.square().sum(dim=-1).mean())
    force_scale = torch.sqrt(training_y.square().mean())
    normalized_x = training_x / displacement_scale
    normalized_y = training_y / force_scale
    validation_x = dataset.displacement[validation_indices] / displacement_scale
    validation_r = dataset.relative_rotation[validation_indices]
    validation_y = dataset.target[validation_indices, 0] / force_scale

    models = compile_models(config.channels)
    linear_parameters = dict(models["linear"].named_parameters())
    svd_parameters = dict(models["svd"].named_parameters())
    if linear_parameters.keys() != svd_parameters.keys():
        raise RuntimeError("model parameter layouts do not match")
    if not all(
        torch.equal(linear_parameters[name], svd_parameters[name])
        for name in linear_parameters
    ):
        raise RuntimeError("model linear initializations do not match")

    histories: dict[str, dict[str, Any]] = {}
    results: dict[str, Any] = {}
    for name, model in models.items():
        pose = model.pose_encoder(training_r).detach()
        validation_pose = model.pose_encoder(validation_r).detach()
        histories[name] = train_model(
            name,
            model,
            normalized_x,
            pose,
            normalized_y,
            validation_x,
            validation_pose,
            validation_y,
            steps=config.steps,
            learning_rate=config.learning_rate,
            force_scale=force_scale,
        )
        results[name] = {
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "training": histories[name],
            "training_evaluation": evaluate(
                model,
                dataset,
                training_evaluation_indices,
                displacement_scale,
                force_scale,
            ),
            "validation": evaluate(
                model,
                dataset,
                validation_indices,
                displacement_scale,
                force_scale,
            ),
            "gradient_probe": gradient_probe(
                model,
                normalized_x[:2],
                training_r[:2],
                normalized_y[:2],
            ),
            "symmetry_residual": symmetry_residual(
                model,
                normalized_x[:2],
                training_r[:2],
            ),
        }

    figure_path = output_directory / "benzene_group_conv_training_loss.png"
    prediction_figure_path = output_directory / "benzene_group_conv_predictions.png"
    record_path = output_directory / "benzene_group_conv_results.json"
    record = {
        "data_file": DATA_FILE,
        "dtype": str(DTYPE),
        "device": "cpu",
        "split_seed": SPLIT_SEED,
        "model_seed": MODEL_SEED,
        "training_indices": training_indices.tolist(),
        "training_evaluation_indices": training_evaluation_indices.tolist(),
        "validation_indices": validation_indices.tolist(),
        "displacement_scale": float(displacement_scale),
        "force_scale": float(force_scale),
        "channels": config.channels,
        "training_count": config.training_count,
        "validation_count": config.validation_count,
        "steps": config.steps,
        "learning_rate": config.learning_rate,
        "results": results,
        "figure_path": str(figure_path),
        "prediction_figure_path": str(prediction_figure_path),
    }
    record_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf_8",
    )
    plot_histories(histories, figure_path)
    plot_predictions(results, prediction_figure_path)

    printable = copy.deepcopy(record)
    for key in (
        "training_indices",
        "training_evaluation_indices",
        "validation_indices",
    ):
        values = printable[key]
        printable[key] = {
            "count": len(values),
            "first_ten": values[:10],
            "last_ten": values[-10:],
        }
    for model_result in printable["results"].values():
        model_result["training"] = {
            "initial_force_mse": model_result["training"]["force_mse"][0],
            "final_force_mse": model_result["training"]["force_mse"][-1],
            "maximum_gradient": model_result["training"]["maximum_gradient"],
            "seconds": model_result["training"]["seconds"],
        }
        model_result["training_evaluation"].pop("samples")
        model_result["validation"].pop("samples")
    printable["record_path"] = str(record_path)
    print(json.dumps(printable, indent=2, ensure_ascii=False))
    return record


def build_parser() -> argparse.ArgumentParser:
    """Build the comparison command line interface."""
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_count", type=int, default=1900)
    parser.add_argument("--validation_count", type=int, default=100)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=0.005)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--output_directory",
        type=Path,
        default=repository_root / "tmp" / "benzene_group_conv_comparison_2000",
    )
    return parser


def main() -> int:
    """Run the configured comparison."""
    config = build_parser().parse_args()
    if config.steps < 1:
        raise ValueError("steps must be positive")
    if config.training_count < config.validation_count:
        raise ValueError("training count must cover the training evaluation")
    if config.validation_count < 1:
        raise ValueError("validation count must be positive")
    if config.learning_rate <= 0.0 or not math.isfinite(config.learning_rate):
        raise ValueError("learning rate must be finite and positive")
    if config.channels != 2:
        raise ValueError("channels must equal two for matched parameter counts")
    if config.threads < 1:
        raise ValueError("threads must be positive")
    run(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
