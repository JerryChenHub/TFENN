"""Benzene force smoke test using only compiled linear intertwiners."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import Tensor, nn

from TFENN.data import BenzenePairDataset
from TFENN.tensor_math import (
    PoseEncoder,
    compile_anchors,
    compile_intertwiners,
)


DTYPE = torch.float64
TRAIN_INDICES = (54, 85)
TRAIN_STEPS = 500


def _benzene_generators() -> Tensor:
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


class TypedIntertwinerLinear(nn.Module):
    """Mix channels through one fixed intertwiner basis."""

    def __init__(
        self,
        basis: Tensor,
        generator: torch.Generator,
        *,
        initialize_identity: bool,
        scale: float = 0.08,
    ) -> None:
        """Register the basis and initialize one input and output channel."""
        super().__init__()
        self.register_buffer("basis", basis.clone())
        self.weight = nn.Parameter(
            scale
            * torch.randn(
                1,
                1,
                basis.shape[0],
                dtype=basis.dtype,
                generator=generator,
            )
        )
        if initialize_identity:
            with torch.no_grad():
                self.weight[0, 0].copy_(torch.einsum("kii->k", basis))

    def forward(self, value: Tensor) -> Tensor:
        """Apply the learned basis mixture without bias or activation."""
        return torch.einsum(
            "ock,kdi,bci->bod",
            self.weight,
            self.basis,
            value,
        )


class BenzeneLinearForceSmoke(nn.Module):
    """Implement the requested A and B linear paths for force only."""

    def __init__(
        self,
        encoder: PoseEncoder,
        lift_aa: Tensor,
        lift_ba: Tensor,
        lift_bb: Tensor,
        *,
        seed: int,
    ) -> None:
        """Build three A blocks, three B blocks, the merge, and force head."""
        super().__init__()
        self.pose_encoder = encoder
        generator = torch.Generator().manual_seed(seed)

        self.a1 = TypedIntertwinerLinear(
            lift_aa, generator, initialize_identity=True
        )
        self.a2 = TypedIntertwinerLinear(
            lift_aa, generator, initialize_identity=True
        )
        self.a3 = TypedIntertwinerLinear(
            lift_aa, generator, initialize_identity=True
        )
        self.b1 = TypedIntertwinerLinear(
            lift_bb, generator, initialize_identity=True
        )
        self.b2 = TypedIntertwinerLinear(
            lift_bb, generator, initialize_identity=True
        )
        self.b3 = TypedIntertwinerLinear(
            lift_bb, generator, initialize_identity=True
        )
        self.a4_from_a = TypedIntertwinerLinear(
            lift_aa, generator, initialize_identity=True
        )
        self.a4_from_b = TypedIntertwinerLinear(
            lift_ba,
            generator,
            initialize_identity=False,
            scale=0.03,
        )
        self.force_head = TypedIntertwinerLinear(
            lift_aa, generator, initialize_identity=True
        )

    def force_from_pose(self, displacement: Tensor, pose: Tensor) -> Tensor:
        """Evaluate the trainable paths from cached A and B inputs."""
        value_a = self.a3(self.a2(self.a1(displacement[:, None])))
        value_b = self.b3(self.b2(self.b1(pose[:, None])))
        value_a4 = self.a4_from_a(value_a) + self.a4_from_b(value_b)
        return self.force_head(value_a4)[:, 0]

    def forward(self, displacement: Tensor, rotation: Tensor) -> Tensor:
        """Encode relative pose and return receiver frame force."""
        return self.force_from_pose(
            displacement,
            self.pose_encoder(rotation),
        )


def _compile_model() -> tuple[BenzeneLinearForceSmoke, Tensor, Tensor]:
    """Compile D6 anchors and all linear bases used by the smoke model."""
    generators = _benzene_generators()
    encoder = PoseEncoder(compile_anchors(generators, output_ranks=(2, 6)))
    representation_b = encoder.representation(generators)
    lift_aa = compile_intertwiners(generators, generators).basis
    lift_ba = compile_intertwiners(representation_b, generators).basis
    lift_bb = compile_intertwiners(representation_b, representation_b).basis
    assert lift_aa.shape == (2, 3, 3)
    assert lift_ba.shape == (4, 3, 18)
    assert lift_bb.shape == (30, 18, 18)
    model = BenzeneLinearForceSmoke(
        encoder,
        lift_aa,
        lift_ba,
        lift_bb,
        seed=20260802,
    )
    return model, lift_aa, lift_ba


def _load_training_pair(repository_root: Path) -> tuple[Tensor, Tensor, Tensor]:
    """Load two well conditioned rows and select force without moment."""
    path = (
        repository_root
        / "data"
        / "benzene_pair"
        / "Benzene_10000_5.0_10.0_3.0_gamma1.csv"
    )
    dataset = BenzenePairDataset(path, dtype=DTYPE)
    indices = torch.tensor(TRAIN_INDICES)
    return (
        dataset.displacement[indices],
        dataset.relative_rotation[indices],
        dataset.target[indices, 0],
    )


def _maximum_generator_residual(
    model: BenzeneLinearForceSmoke,
    displacement: Tensor,
    rotation: Tensor,
) -> float:
    """Measure independent receiver covariance and sender invariance."""
    maximum = 0.0
    identity = torch.eye(3, dtype=displacement.dtype)
    actions = torch.cat((identity[None], _benzene_generators()), dim=0)
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


def test_benzene_force_deep_linear_smoke_overfits(
    repository_root: Path,
) -> None:
    """Overfit two force samples with finite gradients and exact symmetry."""
    model, lift_aa, lift_ba = _compile_model()
    displacement, rotation, force = _load_training_pair(repository_root)

    displacement_scale = torch.sqrt(
        displacement.square().sum(dim=-1).mean()
    )
    force_scale = torch.sqrt(force.square().mean())
    normalized_displacement = displacement / displacement_scale
    normalized_force = force / force_scale
    pose = model.pose_encoder(rotation).detach()

    columns = torch.cat(
        (
            torch.einsum(
                "koi,ni->nko", lift_aa, normalized_displacement
            ),
            torch.einsum("koi,ni->nko", lift_ba, pose),
        ),
        dim=1,
    )
    design = columns.permute(0, 2, 1).reshape(6, 6)
    singular_values = torch.linalg.svdvals(design)
    solution = torch.linalg.solve(design, normalized_force.reshape(6))
    analytic_residual = design @ solution - normalized_force.reshape(6)
    assert int(torch.linalg.matrix_rank(design)) == 6
    assert float(analytic_residual.abs().max()) < 1.0e-12

    probe_x = normalized_displacement.clone().requires_grad_(True)
    probe_r = rotation.clone().requires_grad_(True)
    probe_loss = torch.nn.functional.mse_loss(
        model(probe_x, probe_r),
        normalized_force,
    )
    probe_loss.backward()
    assert probe_x.grad is not None and bool(torch.isfinite(probe_x.grad).all())
    assert probe_r.grad is not None and bool(torch.isfinite(probe_r.grad).all())
    assert float(torch.linalg.vector_norm(probe_x.grad)) > 0.0
    assert float(torch.linalg.vector_norm(probe_r.grad)) > 0.0
    model.zero_grad(set_to_none=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    named_parameters = dict(model.named_parameters())
    initial_loss = float(
        torch.nn.functional.mse_loss(
            model.force_from_pose(normalized_displacement, pose),
            normalized_force,
        ).detach()
    )
    maximum_gradient = 0.0
    minimum_initial_gradient = math.inf
    for step in range(TRAIN_STEPS):
        optimizer.zero_grad(set_to_none=True)
        prediction = model.force_from_pose(normalized_displacement, pose)
        loss = torch.nn.functional.mse_loss(prediction, normalized_force)
        assert bool(torch.isfinite(loss))
        loss.backward()
        named_gradients = {
            name: parameter.grad
            for name, parameter in named_parameters.items()
        }
        assert all(value is not None for value in named_gradients.values())
        gradients = [
            value for value in named_gradients.values() if value is not None
        ]
        assert all(bool(torch.isfinite(value).all()) for value in gradients)
        maximum_gradient = max(
            maximum_gradient,
            max(float(value.detach().abs().max()) for value in gradients),
        )
        if step == 0:
            minimum_initial_gradient = min(
                float(torch.linalg.vector_norm(value.detach()))
                for value in gradients
            )
            assert minimum_initial_gradient > 0.0
            initial_parameters = {
                name: parameter.detach().clone()
                for name, parameter in named_parameters.items()
            }
        optimizer.step()
        if step == 0:
            assert all(
                not torch.equal(initial_parameters[name], parameter.detach())
                for name, parameter in named_parameters.items()
            )

    final_prediction = (
        model.force_from_pose(normalized_displacement, pose).detach() * force_scale
    )
    final_loss = float(
        torch.nn.functional.mse_loss(final_prediction, force).detach()
    )
    maximum_force_error = float((final_prediction - force).abs().max())
    symmetry_residual = _maximum_generator_residual(
        model,
        normalized_displacement,
        rotation,
    )

    assert initial_loss > 1.0e-2
    assert final_loss < 1.0e-16
    assert maximum_force_error < 1.0e-7
    assert 0.0 < minimum_initial_gradient < 10.0
    assert 0.0 < maximum_gradient < 10.0
    assert symmetry_residual < 1.0e-10

    print(
        {
            "indices": TRAIN_INDICES,
            "design_rank": 6,
            "design_condition": float(
                singular_values[0] / singular_values[-1]
            ),
            "initial_normalized_mse": initial_loss,
            "final_force_mse": final_loss,
            "maximum_force_error": maximum_force_error,
            "minimum_initial_gradient": minimum_initial_gradient,
            "maximum_gradient": maximum_gradient,
            "symmetry_residual": symmetry_residual,
        }
    )
