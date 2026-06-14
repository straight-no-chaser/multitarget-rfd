"""
Inference-time multi-state binder guidance for RFdiffusion.

This module adds a conservative geometric feasibility layer on top of the
existing single-target inference loop. It does not modify model weights,
training code, or network inputs beyond choosing a single reference target for
the underlying RFdiffusion run.
"""

import glob
import logging
import math
import os
import pickle
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from rfdiffusion.inference import utils as iu
from rfdiffusion.util import writepdb, writepdb_multi


HOTSPOT_RE = re.compile(r"^([A-Za-z])(-?\d+)$")


@dataclass
class LoadedMultiStateTarget:
    path: str
    name: str
    hotspot_labels: List[str]
    pdb_idx: List[Tuple[str, int]]
    xyz: torch.Tensor
    mask: torch.Tensor
    hotspot_indices: List[int]
    hotspot_ca: torch.Tensor
    hotspot_centroid: torch.Tensor
    hotspot_axis: Optional[torch.Tensor]
    backbone_atoms: torch.Tensor


class MultiStateTargets:
    def __init__(
        self,
        targets: Sequence[LoadedMultiStateTarget],
        reference_target_idx: int,
    ) -> None:
        self.targets = list(targets)
        self.reference_target_idx = reference_target_idx
        self.alignment_report: List[Dict[str, Any]] = []

    @property
    def reference(self) -> LoadedMultiStateTarget:
        return self.targets[self.reference_target_idx]

    @property
    def num_targets(self) -> int:
        return len(self.targets)

    def to(self, device: torch.device) -> "MultiStateTargets":
        for target in self.targets:
            target.xyz = target.xyz.to(device)
            target.mask = target.mask.to(device)
            target.hotspot_ca = target.hotspot_ca.to(device)
            target.hotspot_centroid = target.hotspot_centroid.to(device)
            target.backbone_atoms = target.backbone_atoms.to(device)
            if target.hotspot_axis is not None:
                target.hotspot_axis = target.hotspot_axis.to(device)
        return self


def _make_deterministic(seed: int = 0) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _to_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _parse_hotspot_label(label: str) -> Tuple[str, int]:
    match = HOTSPOT_RE.match(str(label))
    if match is None:
        raise ValueError(
            f"Invalid hotspot residue '{label}'. Expected chain+residue form such as A45."
        )
    chain_id, residue_idx = match.groups()
    return chain_id, int(residue_idx)


def _normalize_hotspot_lists(raw_hotspots: Any, n_targets: int) -> List[List[str]]:
    hotspot_lists = _to_list(raw_hotspots)
    if n_targets == 1 and hotspot_lists and isinstance(hotspot_lists[0], str):
        hotspot_lists = [hotspot_lists]

    normalized: List[List[str]] = []
    for hotspot_group in hotspot_lists:
        if isinstance(hotspot_group, str):
            normalized.append([hotspot_group])
        else:
            normalized.append([str(h) for h in list(hotspot_group)])

    if len(normalized) != n_targets:
        raise ValueError(
            "multistate.hotspot_res_by_target must contain one hotspot list per target "
            f"({len(normalized)} provided for {n_targets} targets)."
        )
    if any(len(group) == 0 for group in normalized):
        raise ValueError("Each multistate target must have at least one hotspot residue.")
    return normalized


def parse_multistate_config(cfg: DictConfig) -> Dict[str, Any]:
    multistate_cfg = OmegaConf.select(cfg, "multistate", default=None)
    if multistate_cfg is None:
        return {"enable": False}

    parsed: Dict[str, Any] = {
        "enable": bool(OmegaConf.select(cfg, "multistate.enable", default=False))
    }
    if not parsed["enable"]:
        return parsed

    target_pdbs = [to_absolute_path(path) for path in _to_list(multistate_cfg.target_pdbs)]
    if len(target_pdbs) < 2:
        raise ValueError(
            "Phase 1 multistate guidance requires at least two pre-aligned target PDBs."
        )
    for pdb_path in target_pdbs:
        if not os.path.exists(pdb_path):
            raise FileNotFoundError(
                f"Could not find multistate target PDB: {pdb_path}"
            )

    hotspot_res_by_target = _normalize_hotspot_lists(
        multistate_cfg.hotspot_res_by_target, len(target_pdbs)
    )

    reference_target_idx = int(multistate_cfg.reference_target_idx)
    if reference_target_idx < 0 or reference_target_idx >= len(target_pdbs):
        raise ValueError(
            "multistate.reference_target_idx is out of range for "
            f"{len(target_pdbs)} targets."
        )

    aggregate_mode = str(multistate_cfg.aggregate_mode)
    if aggregate_mode not in {"softmax_max", "max"}:
        raise ValueError(
            "multistate.aggregate_mode must be 'softmax_max' or 'max', "
            f"got '{aggregate_mode}'."
        )

    guide_schedule = str(multistate_cfg.guide_schedule)
    if guide_schedule not in {"constant", "linear", "mid_peak"}:
        raise ValueError(
            "multistate.guide_schedule must be one of "
            "'constant', 'linear', or 'mid_peak'."
        )

    softmax_temperature = float(multistate_cfg.softmax_temperature)
    if aggregate_mode == "softmax_max" and softmax_temperature <= 0:
        raise ValueError("multistate.softmax_temperature must be > 0 for softmax_max.")

    parsed.update(
        {
            "target_pdbs": target_pdbs,
            "hotspot_res_by_target": hotspot_res_by_target,
            "reference_target_idx": reference_target_idx,
            "reference_target_pdb": target_pdbs[reference_target_idx],
            "reference_hotspots": hotspot_res_by_target[reference_target_idx],
            "aggregate_mode": aggregate_mode,
            "softmax_temperature": softmax_temperature,
            "guide_scale": float(multistate_cfg.guide_scale),
            "guide_schedule": guide_schedule,
            "use_pose_consistency": bool(multistate_cfg.use_pose_consistency),
            "pose_weight": float(multistate_cfg.pose_weight),
            "hotspot_weight": float(multistate_cfg.hotspot_weight),
            "clash_weight": float(multistate_cfg.clash_weight),
            "contact_weight": float(multistate_cfg.contact_weight),
            "contact_cutoff": float(multistate_cfg.contact_cutoff),
            "clash_cutoff": float(multistate_cfg.clash_cutoff),
            "alignment_check": bool(multistate_cfg.alignment_check),
            "centroid_tolerance": float(multistate_cfg.centroid_tolerance),
            "axis_tolerance_deg": float(multistate_cfg.axis_tolerance_deg),
            "grad_clip_norm": float(multistate_cfg.grad_clip_norm),
            "reject_bad_guidance_step": bool(
                multistate_cfg.reject_bad_guidance_step
            ),
        }
    )
    return parsed


def _prepare_reference_target_inputs(
    cfg: DictConfig, multistate_conf: Dict[str, Any], log: logging.Logger
) -> None:
    reference_target_pdb = multistate_conf["reference_target_pdb"]
    reference_hotspots = multistate_conf["reference_hotspots"]

    input_pdb = OmegaConf.select(cfg, "inference.input_pdb", default=None)
    if input_pdb is not None:
        input_pdb = to_absolute_path(str(input_pdb))
        if input_pdb != reference_target_pdb:
            log.warning(
                "multistate.enable=true uses multistate.reference_target_idx as the "
                "RFdiffusion conditioning target. Overriding inference.input_pdb "
                "with %s.",
                reference_target_pdb,
            )
    cfg.inference.input_pdb = reference_target_pdb

    existing_hotspots = OmegaConf.select(cfg, "ppi.hotspot_res", default=None)
    if existing_hotspots is not None:
        existing_hotspots = [str(h) for h in list(existing_hotspots)]
        if existing_hotspots != reference_hotspots:
            log.warning(
                "Overriding ppi.hotspot_res with the reference multistate hotspot "
                "set for consistency: %s",
                reference_hotspots,
            )
    cfg.ppi.hotspot_res = reference_hotspots


def _principal_axis_from_coords(coords: torch.Tensor) -> Optional[torch.Tensor]:
    if coords.shape[0] < 2:
        return None
    centered = coords - coords.mean(dim=0, keepdim=True)
    if torch.linalg.vector_norm(centered) < 1e-6:
        return None
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    axis_norm = torch.linalg.vector_norm(axis)
    if axis_norm < 1e-8 or not torch.isfinite(axis_norm):
        return None
    return axis / axis_norm


def _principal_axis_for_pose(coords: torch.Tensor) -> torch.Tensor:
    axis = _principal_axis_from_coords(coords)
    if axis is not None:
        return axis

    if coords.shape[0] >= 2:
        axis = coords[-1] - coords[0]
    else:
        axis = torch.tensor([1.0, 0.0, 0.0], device=coords.device, dtype=coords.dtype)
    axis_norm = torch.linalg.vector_norm(axis)
    if axis_norm < 1e-8 or not torch.isfinite(axis_norm):
        axis = torch.tensor([1.0, 0.0, 0.0], device=coords.device, dtype=coords.dtype)
        axis_norm = torch.linalg.vector_norm(axis)
    return axis / axis_norm


def _axis_angle_deg(axis_a: torch.Tensor, axis_b: torch.Tensor) -> float:
    cosine = torch.clamp(torch.abs(torch.dot(axis_a, axis_b)), 0.0, 1.0)
    return float(torch.rad2deg(torch.arccos(cosine)).item())


def _flatten_backbone_atoms(xyz: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    backbone_xyz = xyz[:, :4, :]
    backbone_mask = mask[:, :4]
    return backbone_xyz[backbone_mask]


def load_multistate_targets(cfg_or_conf: Any) -> MultiStateTargets:
    multistate_conf = (
        parse_multistate_config(cfg_or_conf)
        if isinstance(cfg_or_conf, DictConfig)
        else cfg_or_conf
    )
    if not multistate_conf.get("enable", False):
        raise ValueError("load_multistate_targets called while multistate is disabled.")

    targets: List[LoadedMultiStateTarget] = []
    for pdb_path, hotspot_labels in zip(
        multistate_conf["target_pdbs"], multistate_conf["hotspot_res_by_target"]
    ):
        parsed = iu.parse_pdb(pdb_path)
        pdb_idx = list(parsed["pdb_idx"])
        hotspot_indices: List[int] = []
        hotspot_lookup = {_parse_hotspot_label(label) for label in hotspot_labels}
        for i, residue_id in enumerate(pdb_idx):
            if residue_id in hotspot_lookup:
                hotspot_indices.append(i)

        if len(hotspot_indices) != len(hotspot_lookup):
            missing = sorted(
                f"{chain}{resnum}"
                for chain, resnum in hotspot_lookup
                if (chain, resnum) not in pdb_idx
            )
            raise ValueError(
                f"Target {pdb_path} is missing hotspot residues: {', '.join(missing)}"
            )

        xyz = torch.from_numpy(parsed["xyz"]).float()
        mask = torch.from_numpy(parsed["mask"]).bool()
        hotspot_ca = xyz[hotspot_indices, 1, :]
        hotspot_centroid = hotspot_ca.mean(dim=0)
        hotspot_axis = _principal_axis_from_coords(hotspot_ca)
        targets.append(
            LoadedMultiStateTarget(
                path=pdb_path,
                name=os.path.basename(pdb_path),
                hotspot_labels=list(hotspot_labels),
                pdb_idx=pdb_idx,
                xyz=xyz,
                mask=mask,
                hotspot_indices=hotspot_indices,
                hotspot_ca=hotspot_ca,
                hotspot_centroid=hotspot_centroid,
                hotspot_axis=hotspot_axis,
                backbone_atoms=_flatten_backbone_atoms(xyz, mask),
            )
        )

    return MultiStateTargets(
        targets=targets,
        reference_target_idx=multistate_conf["reference_target_idx"],
    )


def run_alignment_sanity_check(
    targets: MultiStateTargets,
    centroid_tolerance: float,
    axis_tolerance_deg: float,
    log: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    reference_target = targets.reference
    report: List[Dict[str, Any]] = []
    failures: List[str] = []

    for target_idx, target in enumerate(targets.targets):
        centroid_shift = float(
            torch.linalg.vector_norm(
                target.hotspot_centroid - reference_target.hotspot_centroid
            ).item()
        )
        axis_angle = None
        if (
            reference_target.hotspot_axis is not None
            and target.hotspot_axis is not None
            and target_idx != targets.reference_target_idx
        ):
            axis_angle = _axis_angle_deg(
                target.hotspot_axis, reference_target.hotspot_axis
            )

        entry = {
            "target_idx": target_idx,
            "target_name": target.name,
            "centroid_shift": centroid_shift,
            "axis_angle_deg": axis_angle,
            "is_reference": target_idx == targets.reference_target_idx,
        }
        report.append(entry)

        if centroid_shift > centroid_tolerance:
            failures.append(
                f"{target.name}: hotspot centroid shift {centroid_shift:.2f} A "
                f"exceeds tolerance {centroid_tolerance:.2f} A"
            )
        if axis_angle is not None and axis_angle > axis_tolerance_deg:
            failures.append(
                f"{target.name}: hotspot axis deviation {axis_angle:.2f} deg "
                f"exceeds tolerance {axis_tolerance_deg:.2f} deg"
            )
        if axis_angle is None and log is not None and not entry["is_reference"]:
            log.warning(
                "Skipping hotspot axis comparison for %s because a stable local axis "
                "could not be estimated from the provided hotspots.",
                target.name,
            )

    if failures:
        raise ValueError(
            "Multi-state alignment sanity check failed. Targets must be pre-aligned "
            "before RFdiffusion inference.\n- " + "\n- ".join(failures)
        )

    if log is not None:
        log.info(
            "Multi-state alignment sanity check passed for %d targets.",
            targets.num_targets,
        )
    targets.alignment_report = report
    return report


def compute_hotspot_proximity_loss(
    binder_xyz: torch.Tensor,
    hotspot_ca: torch.Tensor,
    contact_cutoff: float,
) -> torch.Tensor:
    binder_ca = binder_xyz[:, 1, :]
    dists = torch.cdist(binder_ca, hotspot_ca)
    min_dists = dists.min(dim=0).values
    return torch.relu(min_dists - contact_cutoff).pow(2).mean()


def compute_clash_loss(
    binder_xyz: torch.Tensor,
    target_backbone_atoms: torch.Tensor,
    clash_cutoff: float,
) -> torch.Tensor:
    binder_backbone = binder_xyz[:, :4, :]
    binder_backbone_mask = torch.isfinite(binder_backbone).all(dim=-1)
    binder_atoms = binder_backbone[binder_backbone_mask]
    if binder_atoms.numel() == 0 or target_backbone_atoms.numel() == 0:
        return torch.zeros((), device=binder_xyz.device)
    dists = torch.cdist(binder_atoms, target_backbone_atoms)
    overlap = torch.relu(clash_cutoff - dists)
    return overlap.pow(2).mean()


def compute_contact_reward(
    binder_xyz: torch.Tensor,
    hotspot_ca: torch.Tensor,
    contact_cutoff: float,
) -> torch.Tensor:
    binder_ca = binder_xyz[:, 1, :]
    dists = torch.cdist(binder_ca, hotspot_ca)
    min_dists = dists.min(dim=0).values
    return torch.sigmoid((contact_cutoff - min_dists) / 0.75).mean()


def compute_single_state_interface_loss(
    binder_xyz: torch.Tensor,
    target: LoadedMultiStateTarget,
    multistate_conf: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    hotspot_loss = compute_hotspot_proximity_loss(
        binder_xyz, target.hotspot_ca, multistate_conf["contact_cutoff"]
    )
    clash_loss = compute_clash_loss(
        binder_xyz, target.backbone_atoms, multistate_conf["clash_cutoff"]
    )
    contact_reward = compute_contact_reward(
        binder_xyz, target.hotspot_ca, multistate_conf["contact_cutoff"]
    )
    total_loss = (
        multistate_conf["hotspot_weight"] * hotspot_loss
        + multistate_conf["clash_weight"] * clash_loss
        - multistate_conf["contact_weight"] * contact_reward
    )
    return {
        "total_loss": total_loss,
        "hotspot_loss": hotspot_loss,
        "clash_loss": clash_loss,
        "contact_reward": contact_reward,
    }


def compute_pose_consistency_loss(
    binder_xyz: torch.Tensor,
    targets: MultiStateTargets,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    binder_ca = binder_xyz[:, 1, :]
    binder_centroid = binder_ca.mean(dim=0)
    binder_axis = _principal_axis_for_pose(binder_ca)

    offsets = torch.stack(
        [binder_centroid - target.hotspot_centroid for target in targets.targets], dim=0
    )
    offset_mean = offsets.mean(dim=0, keepdim=True)
    centroid_variance = ((offsets - offset_mean) ** 2).mean()

    axis_losses: List[torch.Tensor] = []
    for target in targets.targets:
        if target.hotspot_axis is None:
            continue
        cosine = torch.clamp(
            torch.abs(torch.dot(binder_axis, target.hotspot_axis)), 0.0, 1.0
        )
        axis_losses.append(1.0 - cosine)

    if axis_losses:
        axis_loss = torch.stack(axis_losses).mean()
    else:
        axis_loss = torch.zeros((), device=binder_xyz.device)

    total_pose_loss = centroid_variance + axis_loss
    return total_pose_loss, {
        "centroid_variance": centroid_variance,
        "axis_loss": axis_loss,
    }


def aggregate_multistate_loss(
    losses: torch.Tensor, mode: str, temperature: float = 0.5
) -> torch.Tensor:
    if mode == "softmax_max":
        return temperature * torch.logsumexp(losses / temperature, dim=0)
    if mode == "max":
        return losses.max()
    raise ValueError(f"Unsupported multistate aggregate mode: {mode}")


def get_guide_scale(
    step: int, total_steps: int, mode: str, base_scale: float
) -> float:
    if base_scale <= 0:
        return 0.0
    if total_steps <= 1:
        return float(base_scale)

    progress = 1.0 - float(step - 1) / float(max(total_steps - 1, 1))
    if mode == "constant":
        factor = 1.0
    elif mode == "linear":
        factor = progress
    elif mode == "mid_peak":
        factor = 0.25 + 0.75 * math.sin(math.pi * progress)
    else:
        raise ValueError(f"Unsupported guide schedule: {mode}")
    return float(base_scale) * factor


def _evaluate_multistate_objective(
    xyz: torch.Tensor,
    binderlen: int,
    targets: MultiStateTargets,
    multistate_conf: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    binder_xyz = xyz[:binderlen]
    per_target_metrics = [
        compute_single_state_interface_loss(binder_xyz, target, multistate_conf)
        for target in targets.targets
    ]
    per_target_losses = torch.stack(
        [metrics["total_loss"] for metrics in per_target_metrics], dim=0
    )
    aggregate_interface_loss = aggregate_multistate_loss(
        per_target_losses,
        multistate_conf["aggregate_mode"],
        multistate_conf["softmax_temperature"],
    )

    pose_loss = torch.zeros((), device=xyz.device)
    pose_metrics: Dict[str, torch.Tensor] = {
        "centroid_variance": torch.zeros((), device=xyz.device),
        "axis_loss": torch.zeros((), device=xyz.device),
    }
    if multistate_conf["use_pose_consistency"]:
        pose_loss, pose_metrics = compute_pose_consistency_loss(binder_xyz, targets)

    total_objective = aggregate_interface_loss + (
        multistate_conf["pose_weight"] * pose_loss
    )
    max_clash_loss = torch.stack(
        [metrics["clash_loss"] for metrics in per_target_metrics], dim=0
    ).max()

    return total_objective, {
        "total_objective": total_objective,
        "aggregate_interface_loss": aggregate_interface_loss,
        "pose_loss": pose_loss,
        "max_clash_loss": max_clash_loss,
        "per_target": per_target_metrics,
        "per_target_total_losses": per_target_losses,
        "pose_metrics": pose_metrics,
    }


def _metrics_to_python(metrics: Dict[str, Any]) -> Dict[str, Any]:
    per_target_python: List[Dict[str, float]] = []
    for target_metrics in metrics["per_target"]:
        per_target_python.append(
            {
                key: float(value.detach().cpu().item())
                for key, value in target_metrics.items()
            }
        )

    pose_metrics = {
        key: float(value.detach().cpu().item())
        for key, value in metrics["pose_metrics"].items()
    }
    return {
        "total_objective": float(metrics["total_objective"].detach().cpu().item()),
        "aggregate_interface_loss": float(
            metrics["aggregate_interface_loss"].detach().cpu().item()
        ),
        "pose_loss": float(metrics["pose_loss"].detach().cpu().item()),
        "max_clash_loss": float(metrics["max_clash_loss"].detach().cpu().item()),
        "per_target_total_losses": [
            float(value.detach().cpu().item())
            for value in metrics["per_target_total_losses"]
        ],
        "per_target": per_target_python,
        "pose_metrics": pose_metrics,
    }


def _guidance_step_is_safe(
    current_metrics: Dict[str, Any], candidate_metrics: Dict[str, Any]
) -> bool:
    if not math.isfinite(candidate_metrics["total_objective"]):
        return False
    if candidate_metrics["max_clash_loss"] > current_metrics["max_clash_loss"] + max(
        0.1, 0.5 * current_metrics["max_clash_loss"]
    ):
        return False
    if candidate_metrics["total_objective"] > current_metrics["total_objective"] + 1e-4:
        return False
    return True


def apply_multistate_guidance(
    xyz: torch.Tensor,
    binderlen: int,
    targets: MultiStateTargets,
    multistate_conf: Dict[str, Any],
    step: int,
    total_steps: int,
    diffusion_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    guide_scale = get_guide_scale(
        step, total_steps, multistate_conf["guide_schedule"], multistate_conf["guide_scale"]
    )
    if binderlen <= 0 or guide_scale <= 0:
        return xyz, {"applied": False, "guide_scale": guide_scale}

    update_mask = torch.ones(binderlen, dtype=torch.bool, device=xyz.device)
    if diffusion_mask is not None:
        update_mask = ~diffusion_mask.to(xyz.device).squeeze()[:binderlen]

    if not update_mask.any():
        return xyz, {
            "applied": False,
            "guide_scale": guide_scale,
            "reason": "no_diffused_binder_positions",
        }

    binder_base = xyz[:binderlen].detach()
    binder_ca0 = binder_base[:, 1, :].detach()
    binder_ca = binder_ca0.clone().requires_grad_(True)
    residue_delta = binder_ca - binder_ca0
    guided_binder = binder_base + residue_delta[:, None, :]
    guided_xyz = xyz.clone()
    guided_xyz[:binderlen] = guided_binder

    objective, metrics = _evaluate_multistate_objective(
        guided_xyz, binderlen, targets, multistate_conf
    )
    objective.backward()

    grad = binder_ca.grad
    if grad is None:
        return xyz, {
            "applied": False,
            "guide_scale": guide_scale,
            "reason": "missing_gradient",
        }

    grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    grad[~update_mask] = 0.0

    grad_norm = float(torch.linalg.vector_norm(grad).detach().cpu().item())
    grad_clip_norm = max(float(multistate_conf["grad_clip_norm"]), 1e-8)
    if grad_norm > grad_clip_norm:
        grad = grad * (grad_clip_norm / (grad_norm + 1e-8))

    step_delta = -guide_scale * grad
    if not torch.isfinite(step_delta).all():
        return xyz, {
            "applied": False,
            "guide_scale": guide_scale,
            "reason": "non_finite_step",
        }

    current_metrics = _metrics_to_python(metrics)
    step_norm = float(torch.linalg.vector_norm(step_delta).detach().cpu().item())
    if step_norm <= 0:
        return xyz, {
            "applied": False,
            "guide_scale": guide_scale,
            "reason": "zero_step",
            "pre_guidance": current_metrics,
        }

    backoff_scales = [1.0]
    if multistate_conf["reject_bad_guidance_step"]:
        backoff_scales = [1.0, 0.5, 0.25]

    for backoff in backoff_scales:
        candidate_xyz = xyz.clone()
        candidate_xyz[:binderlen] = candidate_xyz[:binderlen] + (
            step_delta * backoff
        )[:, None, :]
        if not torch.isfinite(candidate_xyz[:binderlen, :4, :]).all():
            continue

        with torch.no_grad():
            _, candidate_metrics_raw = _evaluate_multistate_objective(
                candidate_xyz, binderlen, targets, multistate_conf
            )
        candidate_metrics = _metrics_to_python(candidate_metrics_raw)

        if (
            not multistate_conf["reject_bad_guidance_step"]
            or _guidance_step_is_safe(current_metrics, candidate_metrics)
        ):
            return candidate_xyz, {
                "applied": True,
                "guide_scale": guide_scale,
                "backoff_scale": backoff,
                "grad_norm": grad_norm,
                "step_norm": step_norm * backoff,
                "pre_guidance": current_metrics,
                "post_guidance": candidate_metrics,
            }

    return xyz, {
        "applied": False,
        "guide_scale": guide_scale,
        "reason": "rejected_bad_guidance_step",
        "grad_norm": grad_norm,
        "step_norm": step_norm,
        "pre_guidance": current_metrics,
    }


def _design_startnum(sampler: Any) -> int:
    design_startnum = sampler.inf_conf.design_startnum
    if design_startnum == -1:
        existing = glob.glob(sampler.inf_conf.output_prefix + "*.pdb")
        indices = [-1]
        for pdb_path in existing:
            match = re.match(r".*_(\d+)\.pdb$", pdb_path)
            if match is None:
                continue
            indices.append(int(match.groups()[0]))
        design_startnum = max(indices) + 1
    return design_startnum


def run_multistate_inference_wrapper(cfg: DictConfig) -> None:
    log = logging.getLogger(__name__)
    multistate_conf = parse_multistate_config(cfg)
    if not multistate_conf.get("enable", False):
        raise ValueError(
            "run_multistate_inference_wrapper was called without multistate.enable=true."
        )

    _prepare_reference_target_inputs(cfg, multistate_conf, log)
    targets = load_multistate_targets(multistate_conf)
    if multistate_conf["alignment_check"]:
        run_alignment_sanity_check(
            targets,
            multistate_conf["centroid_tolerance"],
            multistate_conf["axis_tolerance_deg"],
            log=log,
        )

    if cfg.inference.deterministic:
        _make_deterministic()

    sampler = iu.sampler_selector(cfg)
    targets.to(sampler.device)
    design_startnum = _design_startnum(sampler)

    for i_des in range(design_startnum, design_startnum + sampler.inf_conf.num_designs):
        if cfg.inference.deterministic:
            _make_deterministic(i_des)

        start_time = time.time()
        out_prefix = f"{sampler.inf_conf.output_prefix}_{i_des}"
        log.info("Making design %s", out_prefix)
        if sampler.inf_conf.cautious and os.path.exists(out_prefix + ".pdb"):
            log.info(
                "(cautious mode) Skipping this design because %s.pdb already exists.",
                out_prefix,
            )
            continue

        x_init, seq_init = sampler.sample_init()
        if getattr(sampler, "binderlen", 0) <= 0:
            raise ValueError(
                "Multi-state guidance is inference-only binder guidance and requires "
                "a binder design setup with binderlen > 0."
            )

        denoised_xyz_stack = []
        px0_xyz_stack = []
        seq_stack = []
        plddt_stack = []
        multistate_trace: List[Dict[str, Any]] = []

        x_t = torch.clone(x_init)
        seq_t = torch.clone(seq_init)
        total_steps = int(sampler.t_step_input) - sampler.inf_conf.final_step + 1

        for t in range(
            int(sampler.t_step_input), sampler.inf_conf.final_step - 1, -1
        ):
            px0, x_t, seq_t, plddt = sampler.sample_step(
                t=t, x_t=x_t, seq_init=seq_t, final_step=sampler.inf_conf.final_step
            )
            x_t, guidance_metrics = apply_multistate_guidance(
                x_t,
                sampler.binderlen,
                targets,
                multistate_conf,
                step=t - sampler.inf_conf.final_step + 1,
                total_steps=total_steps,
                diffusion_mask=sampler.mask_str.squeeze(),
            )
            guidance_metrics["timestep"] = t
            multistate_trace.append(guidance_metrics)

            if guidance_metrics.get("applied", False):
                log.info(
                    "Timestep %d multistate guidance applied: scale=%.4f backoff=%.2f "
                    "objective %.4f -> %.4f",
                    t,
                    guidance_metrics["guide_scale"],
                    guidance_metrics["backoff_scale"],
                    guidance_metrics["pre_guidance"]["total_objective"],
                    guidance_metrics["post_guidance"]["total_objective"],
                )
            else:
                log.info(
                    "Timestep %d multistate guidance skipped: %s",
                    t,
                    guidance_metrics.get("reason", "no_update"),
                )

            px0_xyz_stack.append(px0)
            denoised_xyz_stack.append(x_t)
            seq_stack.append(seq_t)
            plddt_stack.append(plddt[0])

        denoised_xyz_stack = torch.flip(torch.stack(denoised_xyz_stack), [0])
        px0_xyz_stack = torch.flip(torch.stack(px0_xyz_stack), [0])
        plddt_stack = torch.stack(plddt_stack)

        os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
        final_seq = torch.where(
            torch.argmax(seq_init, dim=-1) == 21, 7, torch.argmax(seq_init, dim=-1)
        )

        bfacts = torch.ones_like(final_seq.squeeze())
        bfacts[torch.where(torch.argmax(seq_init, dim=-1) == 21, True, False)] = 0

        writepdb(
            f"{out_prefix}.pdb",
            denoised_xyz_stack[0, :, :4],
            final_seq,
            sampler.binderlen,
            chain_idx=sampler.chain_idx,
            bfacts=bfacts,
            idx_pdb=sampler.idx_pdb,
        )

        trb = dict(
            config=OmegaConf.to_container(sampler._conf, resolve=True),
            plddt=plddt_stack.cpu().numpy(),
            device=torch.cuda.get_device_name(torch.cuda.current_device())
            if torch.cuda.is_available()
            else "CPU",
            time=time.time() - start_time,
            multistate={
                "enabled": True,
                "config": multistate_conf,
                "alignment_report": targets.alignment_report,
                "trace": multistate_trace,
            },
        )
        if hasattr(sampler, "contig_map"):
            for key, value in sampler.contig_map.get_mappings().items():
                trb[key] = value
        with open(f"{out_prefix}.trb", "wb") as f_out:
            pickle.dump(trb, f_out)

        if sampler.inf_conf.write_trajectory:
            traj_prefix = (
                os.path.dirname(out_prefix) + "/traj/" + os.path.basename(out_prefix)
            )
            os.makedirs(os.path.dirname(traj_prefix), exist_ok=True)

            writepdb_multi(
                f"{traj_prefix}_Xt-1_traj.pdb",
                denoised_xyz_stack,
                bfacts,
                final_seq.squeeze(),
                use_hydrogens=False,
                backbone_only=False,
                chain_ids=sampler.chain_idx,
            )
            writepdb_multi(
                f"{traj_prefix}_pX0_traj.pdb",
                px0_xyz_stack,
                bfacts,
                final_seq.squeeze(),
                use_hydrogens=False,
                backbone_only=False,
                chain_ids=sampler.chain_idx,
            )

        log.info("Finished design in %.2f minutes", (time.time() - start_time) / 60.0)
