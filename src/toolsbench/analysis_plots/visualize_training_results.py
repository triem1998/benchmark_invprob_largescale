"""
Visualization script for training benchmark results.

Reads parquet files from benchmark outputs and creates Plotly visualizations
for training performance metrics.

Figures generated
-----------------
1. Loss vs Epoch        – train loss per config; if single config also val loss
2. PSNR vs Epoch        – val PSNR per config; if single config also train PSNR
3. Training Time        – stacked bar (fwd / bwd / other) or total per config
                          averaged over epochs; if single config also adds val time
4. GPU Max Memory       – max allocated GPU memory per config

Config labels are sorted by: n_gpu → distribute_model → patch_size →
max_batch_size → checkpoint_batches.

"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.colors
import plotly.graph_objects as go

# Default Plotly qualitative palette
_PALETTE = plotly.colors.qualitative.Plotly


def build_color_map(labels: list[str]) -> dict[str, str]:
    """Assign a consistent color to each config label."""
    return {label: _PALETTE[i % len(_PALETTE)] for i, label in enumerate(labels)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ngpu(row) -> int:
    gres = row.get("p_solver_slurm_gres", "gpu:1")
    try:
        gpus_per_node = int(str(gres).split(":")[-1])
    except (ValueError, IndexError):
        gpus_per_node = 1
    try:
        nodes = int(row.get("p_solver_slurm_nodes", 1) or 1)
    except (ValueError, TypeError):
        nodes = 1
    return gpus_per_node * nodes


def _patch_str(val) -> str:
    """Compact representation of scalar or array patch size."""
    if val is None:
        return "0"
    if isinstance(val, (list, tuple, np.ndarray)):
        unique = list(dict.fromkeys(int(v) for v in val))
        return str(unique[0]) if len(unique) == 1 else str(list(unique))
    return str(int(val))


def create_config_label(row, show_patch: bool = True) -> str:
    """Build a human-readable label for a solver configuration."""
    ngpu = _ngpu(row)
    dist = row.get("p_solver_distribute_model", False)
    patch = _patch_str(row.get("p_solver_patch_size", 0))
    mb = int(row.get("p_solver_max_batch_size", 0) or 0)
    ckpt = row.get("p_solver_checkpoint_batches", "never")

    parts = [f"{ngpu} GPU{'s' if ngpu > 1 else ''}"]
    if dist:
        if show_patch:
            parts.append(f"Patch={patch}")
        parts.append(f"Batch={mb}")
        parts.append(f"Ckpt={ckpt}")
    else:
        parts.append("No Dist")
    return " | ".join(parts)


def _sort_key(row):
    ngpu = _ngpu(row)
    dist = int(bool(row.get("p_solver_distribute_model", False)))
    patch_val = row.get("p_solver_patch_size", 0)
    if isinstance(patch_val, (list, tuple, np.ndarray)):
        patch_num = int(patch_val[0]) if len(patch_val) > 0 else 0
    else:
        patch_num = int(patch_val or 0)
    mb = int(row.get("p_solver_max_batch_size", 0) or 0)
    # checkpoint_batches order: never < always < auto
    ckpt_order = {"never": 0, "always": 1, "auto": 2}
    ckpt = ckpt_order.get(str(row.get("p_solver_checkpoint_batches", "never")), 3)
    return (ngpu, dist, patch_num, mb, ckpt)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def read_parquet_data(output_dir: str | Path) -> tuple[pd.DataFrame, str]:
    output_path = Path(output_dir)
    parquet_files = list(output_path.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet file found in {output_dir}")

    df = pd.read_parquet(parquet_files[0])

    # Compute sort key columns
    df["_ngpu"] = df.apply(_ngpu, axis=1)
    df["_dist"] = df["p_solver_distribute_model"].astype(bool).astype(int)

    def _patch_num(val):
        if isinstance(val, (list, tuple, np.ndarray)):
            return int(val[0]) if len(val) > 0 else 0
        return int(val or 0)

    df["_patch"] = df["p_solver_patch_size"].apply(_patch_num)
    df["_mb"] = df["p_solver_max_batch_size"].fillna(0).astype(int)
    _ckpt_order = {"never": 0, "always": 1, "auto": 2}
    df["_ckpt"] = (
        df["p_solver_checkpoint_batches"].map(_ckpt_order).fillna(3).astype(int)
    )

    # Show patch in label only when patch size varies among distributed configs
    dist_patches = df.loc[df["_dist"] == 1, "_patch"]
    show_patch = dist_patches.nunique() > 1

    df["config_label"] = df.apply(
        lambda r: create_config_label(r, show_patch=show_patch), axis=1
    )
    df.sort_values(
        by=["_ngpu", "_dist", "_patch", "_mb", "_ckpt", "stop_val"], inplace=True
    )
    df.reset_index(drop=True, inplace=True)

    result_name = output_path.name

    print(f"Loaded: {parquet_files[0].name}  shape={df.shape}")
    print("Configurations found:")
    for label in df["config_label"].unique():
        n = (df["config_label"] == label).sum()
        print(f"  {label!r:55s}  {n} rows")

    return df, result_name


# ---------------------------------------------------------------------------
# Sorted unique config labels helper
# ---------------------------------------------------------------------------


def sorted_config_labels(df: pd.DataFrame) -> list[str]:
    return (
        df[["config_label", "_ngpu", "_dist", "_patch", "_mb", "_ckpt"]]
        .drop_duplicates()
        .sort_values(by=["_ngpu", "_dist", "_patch", "_mb", "_ckpt"])["config_label"]
        .tolist()
    )


def _val_color(config_color: str, color_map: dict[str, str]) -> str:
    """Return a contrasting color for the val trace when there is a single config."""
    for c in _PALETTE:
        if c != config_color:
            return c
    return _PALETTE[1]  # fallback


# ---------------------------------------------------------------------------
# Figure 1 – Loss vs Epoch
# ---------------------------------------------------------------------------


def plot_loss_vs_epoch(
    df: pd.DataFrame, output_dir: Path, color_map: dict[str, str]
) -> go.Figure:
    labels = sorted_config_labels(df)
    single = len(labels) == 1

    # Keep only rows with valid training loss
    train_df = df[df["objective_train_loss"].notna() & (df["stop_val"] >= 1)]

    fig = go.Figure()
    groups = train_df.groupby("config_label")

    for label in labels:
        if label not in groups.groups:
            continue
        g = groups.get_group(label).sort_values("stop_val")
        trace_name = f"{label} – Train" if single else label
        color = color_map[label]
        fig.add_trace(
            go.Scatter(
                x=g["stop_val"],
                y=g["objective_train_loss"],
                mode="lines+markers",
                name=trace_name,
                line=dict(width=3, color=color),
                marker=dict(size=8, color=color),
                hovertemplate=(
                    f"<b>{label}</b><br>Epoch: %{{x}}<br>Train Loss: %{{y:.4f}}<extra></extra>"
                ),
            )
        )
        if single and "objective_val_loss" in g.columns:
            val_data = g[g["objective_val_loss"].notna()]
            if not val_data.empty:
                val_color = _val_color(color, color_map)
                fig.add_trace(
                    go.Scatter(
                        x=val_data["stop_val"],
                        y=val_data["objective_val_loss"],
                        mode="lines+markers",
                        name=f"{label} – Val",
                        line=dict(width=3, dash="dash", color=val_color),
                        marker=dict(size=8, color=val_color),
                        hovertemplate=(
                            f"<b>{label}</b><br>Epoch: %{{x}}<br>Val Loss: %{{y:.4f}}<extra></extra>"
                        ),
                    )
                )

    title_text = (
        "Training Loss vs Epoch" if single else "Training Loss vs Epoch (Train)"
    )
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=20)),
        xaxis_title="Epoch",
        yaxis=dict(
            title="Loss",
            exponentformat="e",
            showexponent="all",
        ),
        template="plotly_white",
        font=dict(size=14),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        margin=dict(l=50, r=20, t=60, b=80),
    )

    out = output_dir / "loss_vs_epoch.html"
    fig.write_html(str(out))
    print(f"Saved: {out}")
    return fig


# ---------------------------------------------------------------------------
# Figure 2 – PSNR vs Epoch
# ---------------------------------------------------------------------------


def plot_psnr_vs_epoch(
    df: pd.DataFrame, output_dir: Path, color_map: dict[str, str]
) -> go.Figure:
    labels = sorted_config_labels(df)
    single = len(labels) == 1

    # Val PSNR is available from stop_val=0 (initial eval) onwards
    val_df = df[df["objective_val_psnr"].notna()]

    fig = go.Figure()
    groups_val = val_df.groupby("config_label")
    groups_train = df[
        df["objective_train_psnr"].notna() & (df["objective_train_psnr"] != 0)
    ].groupby("config_label")

    for label in labels:
        color = color_map[label]
        if label in groups_val.groups:
            g = groups_val.get_group(label).sort_values("stop_val")
            trace_name = f"{label} – Val" if single else label
            fig.add_trace(
                go.Scatter(
                    x=g["stop_val"],
                    y=g["objective_val_psnr"],
                    mode="lines+markers",
                    name=trace_name,
                    line=dict(width=3, color=color),
                    marker=dict(size=8, color=color),
                    hovertemplate=(
                        f"<b>{label}</b><br>Epoch: %{{x}}<br>Val PSNR: %{{y:.2f}} dB<extra></extra>"
                    ),
                )
            )

        if single and label in groups_train.groups:
            g = groups_train.get_group(label).sort_values("stop_val")
            train_color = _val_color(color, color_map)
            fig.add_trace(
                go.Scatter(
                    x=g["stop_val"],
                    y=g["objective_train_psnr"],
                    mode="lines+markers",
                    name=f"{label} – Train",
                    line=dict(width=3, dash="dash", color=train_color),
                    marker=dict(size=8, color=train_color),
                    hovertemplate=(
                        f"<b>{label}</b><br>Epoch: %{{x}}<br>Train PSNR: %{{y:.2f}} dB<extra></extra>"
                    ),
                )
            )

    title_psnr = "PSNR vs Epoch" if single else "PSNR vs Epoch (Val)"
    fig.update_layout(
        title=dict(text=title_psnr, font=dict(size=20)),
        xaxis_title="Epoch",
        yaxis_title="PSNR (dB)",
        template="plotly_white",
        font=dict(size=14),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        margin=dict(l=50, r=20, t=60, b=80),
    )

    out = output_dir / "psnr_vs_epoch.html"
    fig.write_html(str(out))
    print(f"Saved: {out}")
    return fig


# ---------------------------------------------------------------------------
# Figure 3 – Training Time
# ---------------------------------------------------------------------------


def plot_training_time(
    df: pd.DataFrame, output_dir: Path, color_map: dict[str, str], stacked: bool = False
) -> go.Figure:
    labels = sorted_config_labels(df)
    single = len(labels) == 1

    # Use only rows with training data (stop_val >= 1)
    train_df = df[df["objective_train_total_time"].notna() & (df["stop_val"] >= 1)]

    agg = (
        train_df.groupby("config_label")
        .agg(
            mean_total=("objective_train_total_time", "mean"),
            mean_fwd=("objective_mean_fwd_time", "mean"),
            mean_bwd=("objective_mean_bwd_time", "mean"),
            mean_other=("objective_mean_other_time", "mean"),
            mean_val=("objective_val_total_time", "mean"),
            mean_val_per_sample=("objective_val_per_sample_time", "mean"),
            _ngpu=("_ngpu", "first"),
            _dist=("_dist", "first"),
            _patch=("_patch", "first"),
            _mb=("_mb", "first"),
            _ckpt=("_ckpt", "first"),
        )
        .reset_index()
        .sort_values(by=["_ngpu", "_dist", "_patch", "_mb", "_ckpt"])
    )

    # mean_fwd / mean_bwd / mean_other are per-step averages – use them directly.
    agg["mean_fwd"] = agg["mean_fwd"].fillna(0)
    agg["mean_bwd"] = agg["mean_bwd"].fillna(0)
    agg["mean_other"] = agg["mean_other"].fillna(0)
    agg["_step_total"] = agg["mean_fwd"] + agg["mean_bwd"] + agg["mean_other"]

    # Identify reference config: 1 GPU, no distribution (if present)
    ref_mask = (agg["_ngpu"] == 1) & (agg["_dist"] == 0)
    has_ref = bool(ref_mask.any())
    ref_total = float(agg.loc[ref_mask, "mean_total"].iloc[0]) if has_ref else None
    ref_step = float(agg.loc[ref_mask, "_step_total"].iloc[0]) if has_ref else None

    # Use speedup/relative-time only in multi-config scenarios
    use_speedup = has_ref and not single

    # Segment definitions: (key, display name, column, alpha, pattern)
    _segments = [
        ("fwd", "Forward", "mean_fwd", 1.0, ""),
        ("bwd", "Backward", "mean_bwd", 0.65, "/"),
        ("other", "Other", "mean_other", 0.35, "x"),
    ]

    fig = go.Figure()

    if stacked:

        def _apply_alpha(hex_color: str, alpha: float) -> str:
            """Convert #rrggbb to rgba(r,g,b,alpha)."""
            hex_color = hex_color.lstrip("#")
            r, g, b = (
                int(hex_color[0:2], 16),
                int(hex_color[2:4], 16),
                int(hex_color[4:6], 16),
            )
            return f"rgba({r},{g},{b},{alpha:.2f})"

        for seg_key, seg_name, seg_col, seg_alpha, seg_pat in _segments:
            is_last_seg = seg_key == _segments[-1][0]
            for i, row in agg.iterrows():
                lbl = row["config_label"]
                show_leg = lbl == agg.iloc[0]["config_label"]
                base_color = color_map[lbl]
                fill_color = _apply_alpha(base_color, seg_alpha)
                actual_seg = float(row[seg_col])
                actual_total = float(row["_step_total"])
                if use_speedup and ref_step and ref_step > 0:
                    y_val = actual_seg / ref_step
                    y_tot = actual_total / ref_step
                    if is_last_seg:
                        hover = (
                            f"<b>{lbl}</b><br>{seg_name}: {actual_seg:.4f}s/image"
                            f" (rel: %{{y:.3f}})"
                            f"<br>Total: {actual_total:.4f}s/image"
                            f" (rel: {y_tot:.3f})<extra></extra>"
                        )
                    else:
                        hover = (
                            f"<b>{lbl}</b><br>{seg_name}: {actual_seg:.4f}s/image"
                            f" (rel: %{{y:.3f}})<extra></extra>"
                        )
                else:
                    y_val = actual_seg
                    if is_last_seg:
                        hover = (
                            f"<b>{lbl}</b><br>{seg_name}: %{{y:.4f}}s/image"
                            f"<br>Total: {actual_total:.4f}s/image<extra></extra>"
                        )
                    else:
                        hover = f"<b>{lbl}</b><br>{seg_name}: %{{y:.4f}}s/image<extra></extra>"
                fig.add_trace(
                    go.Bar(
                        name=seg_name,
                        x=[lbl],
                        y=[y_val],
                        marker=dict(
                            color=fill_color,
                            pattern=dict(
                                shape=seg_pat,
                                solidity=0.5,
                                fgcolor=_apply_alpha(
                                    base_color, min(1.0, seg_alpha + 0.3)
                                ),
                            ),
                            line=dict(width=1, color=base_color),
                        ),
                        width=0.5,
                        legendgroup=seg_key,
                        showlegend=show_leg,
                        hovertemplate=hover,
                    )
                )
        if use_speedup:
            fig.add_hline(
                y=1.0,
                line_dash="dash",
                line_color="gray",
                line_width=1,
                annotation_text="reference",
                annotation_position="top right",
            )
        barmode = "stack"
        legend_cfg = dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
        )
    else:
        bar_width = 0.3 if single else 0.5
        for row in agg.itertuples():
            lbl = row.config_label
            actual_time = float(row.mean_total)
            if use_speedup and ref_total and ref_total > 0:
                y_val = ref_total / actual_time  # speedup ×
                bar_text = f"×{y_val:.2f}"
                hover = (
                    f"<b>{lbl}</b><br>Speedup: ×%{{y:.2f}}"
                    f"<br>Total time: {actual_time:.2f}s/epoch<extra></extra>"
                )
            else:
                y_val = actual_time
                bar_text = f"{actual_time:.1f}s"
                hover = f"<b>{lbl}</b><br>Training: %{{y:.3f}}s<extra></extra>"
            fig.add_trace(
                go.Bar(
                    name=lbl,
                    x=[lbl],
                    y=[y_val],
                    marker_color=color_map[lbl],
                    text=[bar_text],
                    textposition="outside",
                    textfont=dict(size=12),
                    width=bar_width,
                    showlegend=False,
                    hovertemplate=hover,
                )
            )
        if use_speedup:
            fig.add_hline(
                y=1.0,
                line_dash="dash",
                line_color="gray",
                line_width=1,
                annotation_text="reference (×1)",
                annotation_position="top right",
            )
        barmode = "group"
        legend_cfg = dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
        )

    if single:
        val_rows = agg[agg["mean_val_per_sample"].notna()]
        if not val_rows.empty:
            for row in val_rows.itertuples():
                lbl = row.config_label
                vc = _val_color(color_map[lbl], color_map)
                x_val = "Validation" if stacked else lbl
                val_abs = float(row.mean_val_per_sample)
                if stacked and use_speedup and ref_step and ref_step > 0:
                    val_y = val_abs / ref_step
                    val_txt = f"{val_y:.3f}"
                    v_hover = f"Validation: {val_abs:.4f}s/image (rel: {val_y:.3f})<extra></extra>"
                else:
                    val_y = val_abs
                    val_txt = f"{val_abs:.4f}s"
                    v_hover = "Validation: %{y:.4f}s/image<extra></extra>"
                fig.add_trace(
                    go.Bar(
                        name="Validation",
                        x=[x_val],
                        y=[val_y],
                        marker_color=vc,
                        text=[val_txt],
                        textposition="auto",
                        textfont=dict(size=12),
                        width=0.5 if stacked else bar_width,
                        hovertemplate=v_hover,
                    )
                )

    if stacked:
        if use_speedup and ref_step and ref_step > 0:
            y_title = "Relative time (reference = 1)"
            rel_max = (agg["_step_total"] / ref_step).max()
            val_max = (
                (agg["mean_val_per_sample"].max() / ref_step)
                if agg["mean_val_per_sample"].notna().any()
                else 0
            )
            y_range_top = max(rel_max, val_max) * 1.25
        else:
            y_title = "Avg Time per Image (s)"
            step_max = agg["_step_total"].max()
            val_max = (
                agg["mean_val_per_sample"].max()
                if agg["mean_val_per_sample"].notna().any()
                else 0
            )
            y_range_top = max(step_max, val_max) * 1.25
    else:
        if use_speedup and ref_total and ref_total > 0:
            y_title = "Speedup (×)"
            speedup_max = (ref_total / agg["mean_total"]).max()
            y_range_top = speedup_max * 1.25
        else:
            y_title = "Avg Time per Epoch (s)"
            val_max = (
                agg["mean_val_per_sample"].max()
                if single and agg["mean_val_per_sample"].notna().any()
                else 0
            )
            y_range_top = max(agg["mean_total"].max(), val_max) * 1.25
    title_text = (
        "Average Time per Image"
        if stacked
        else (
            "Speedup vs Reference" if use_speedup else "Average Training Time per Epoch"
        )
    )
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=20)),
        xaxis_title="Configuration",
        yaxis=dict(title=y_title, rangemode="tozero", range=[0, y_range_top]),
        xaxis=dict(tickfont=dict(size=12)),
        barmode=barmode,
        template="plotly_white",
        font=dict(size=14),
        legend=legend_cfg,
        margin=dict(l=50, r=20, t=80, b=60),
        hovermode="x unified",
    )

    out = output_dir / "training_time.html"
    fig.write_html(str(out))
    print(f"Saved: {out}")
    return fig


# ---------------------------------------------------------------------------
# Figure 4 – GPU Max Memory
# ---------------------------------------------------------------------------


def plot_gpu_memory(
    df: pd.DataFrame, output_dir: Path, color_map: dict[str, str]
) -> go.Figure:
    mem_col = "objective_gpu_memory_max_allocated_mb"
    mem_df = df[df[mem_col].notna()]

    if mem_df.empty:
        print("Warning: no GPU memory data found – skipping Figure 4")
        return go.Figure()

    agg = (
        mem_df.groupby("config_label")
        .agg(
            mean_mem=(mem_col, "mean"),
            max_mem=(mem_col, "max"),
            _ngpu=("_ngpu", "first"),
            _dist=("_dist", "first"),
            _patch=("_patch", "first"),
            _mb=("_mb", "first"),
            _ckpt=("_ckpt", "first"),
        )
        .reset_index()
        .sort_values(by=["_ngpu", "_dist", "_patch", "_mb", "_ckpt"])
    )

    fig = go.Figure()
    for row in agg.itertuples():
        lbl = row.config_label
        fig.add_trace(
            go.Bar(
                name=lbl,
                x=[lbl],
                y=[row.mean_mem],
                marker_color=color_map[lbl],
                text=[f"{row.mean_mem:.0f} MB"],
                textposition="auto",
                textfont=dict(size=12),
                width=0.5,
                showlegend=False,
                hovertemplate=f"<b>{lbl}</b><br>GPU Max Mem: %{{y:.1f}} MB<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(text="GPU Max Memory Allocated per Config", font=dict(size=20)),
        xaxis_title="Configuration",
        yaxis_title="GPU Max Memory (MB)",
        xaxis=dict(tickfont=dict(size=12)),
        template="plotly_white",
        font=dict(size=14),
        showlegend=False,
        margin=dict(l=50, r=20, t=60, b=60),
    )

    out = output_dir / "gpu_memory.html"
    fig.write_html(str(out))
    print(f"Saved: {out}")
    return fig


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


def create_dashboard(results_path: Path, result_name: str) -> Path | None:
    required = [
        "loss_vs_epoch.html",
        "psnr_vs_epoch.html",
        "training_time.html",
        "gpu_memory.html",
    ]
    for f in required:
        if not (results_path / f).exists():
            print(f"Warning: {f} not found – skipping dashboard")
            return None

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Training Benchmark Dashboard – {result_name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        html, body {{ width: 100%; height: 100%; overflow-x: hidden; background: white; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
        .header {{
            background: linear-gradient(135deg, #1a6b3a 0%, #2a9d5c 100%);
            color: white; padding: 20px; text-align: center;
        }}
        .header h1 {{ font-size: clamp(1.4em, 3.5vw, 2.2em); margin-bottom: 6px; font-weight: 700; }}
        .header p  {{ font-size: clamp(0.85em, 1.8vw, 1.1em); opacity: 0.9; }}
        .timestamp {{
            text-align: center; padding: 8px;
            background: #f8f9fa; color: #6c757d;
            font-size: clamp(0.75em, 1.4vw, 0.85em);
        }}
        .grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0; width: 100%;
        }}
        .cell {{
            position: relative; width: 100%;
            padding-bottom: 90%;
            background: white; border: 1px solid #e0e0e0;
        }}
        iframe {{
            position: absolute; top: 0; left: 0;
            width: 100%; height: 100%; border: none;
        }}
        .footer {{
            text-align: center; padding: 12px;
            background: #f8f9fa; color: #6c757d;
            font-size: clamp(0.75em, 1.4vw, 0.85em);
        }}
        @media (max-width: 768px) {{
            .grid {{ grid-template-columns: 1fr; }}
            .cell {{ padding-bottom: 85%; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Training Benchmark Dashboard</h1>
        <p>{result_name.replace("_", " ").title()}</p>
    </div>
    <div class="timestamp">Generated {datetime.now().strftime("%B %d, %Y at %H:%M:%S")}</div>
    <div class="grid">
        <div class="cell"><iframe src="loss_vs_epoch.html"  title="Loss vs Epoch"></iframe></div>
        <div class="cell"><iframe src="psnr_vs_epoch.html"  title="PSNR vs Epoch"></iframe></div>
        <div class="cell"><iframe src="training_time.html"  title="Training Time"></iframe></div>
        <div class="cell"><iframe src="gpu_memory.html"     title="GPU Memory"></iframe></div>
    </div>
    <div class="footer">Training Benchmark Analysis</div>
</body>
</html>"""

    out = results_path / "dashboard.html"
    out.write_text(html)
    print(f"Dashboard: {out}")
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def visualize_training_results(
    output_dir: str,
    results_dir: str = "docs/source/_static/images",
    stacked: bool = True,
) -> None:
    print(f"\nReading data from: {output_dir}")
    print("-" * 60)
    df, result_name = read_parquet_data(output_dir)

    results_path = Path(results_dir) / result_name
    results_path.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to: {results_path}")
    print("-" * 60)

    color_map = build_color_map(sorted_config_labels(df))

    plot_loss_vs_epoch(df, results_path, color_map)
    plot_psnr_vs_epoch(df, results_path, color_map)
    plot_training_time(df, results_path, color_map, stacked=stacked)
    plot_gpu_memory(df, results_path, color_map)

    print("-" * 60)
    create_dashboard(results_path, result_name)
    print("-" * 60)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize training benchmark results from a parquet file"
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="results/urban_img_size_1024",
        help="Directory containing the *.parquet benchmark output",
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        default="docs/source/_static/images",
        help="Directory to save HTML visualizations",
    )
    parser.add_argument(
        "--no-stacked",
        dest="stacked",
        action="store_false",
        help="Show total bar only in the training-time chart (default: stacked fwd/bwd/other)",
    )
    parser.set_defaults(stacked=True)
    args = parser.parse_args()
    visualize_training_results(args.output_dir, args.results_dir, stacked=args.stacked)
