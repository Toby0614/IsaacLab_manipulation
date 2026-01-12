#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""2D Sensitivity Map Visualization for Dropout Robustness Analysis.

Generates publication-quality heatmaps showing:
- Success rate vs (Phase/Onset Time) × Duration
- Comparison across M1-M4 policies
- Phase-wise and time-wise sensitivity analysis

Based on poe2.pdf recommendations for:
- Heatmap: success vs (phase or onset time) × duration
- Model comparison visualization

Usage:
    # Generate all visualizations from evaluation results
    python scripts/visualize_sensitivity_map.py \
        --results_dir results/dropout_eval \
        --output_dir results/figures

    # Generate for specific variant
    python scripts/visualize_sensitivity_map.py \
        --results_dir results/dropout_eval \
        --variant 1 \
        --output_dir results/figures

    # Custom colormap and style
    python scripts/visualize_sensitivity_map.py \
        --results_dir results/dropout_eval \
        --colormap "RdYlGn" \
        --style "paper"
"""

import argparse
import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Matplotlib configuration for publication-quality figures
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

# Optional seaborn for enhanced styling
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def parse_args():
    parser = argparse.ArgumentParser(description="Generate 2D sensitivity maps from dropout evaluation results")
    
    parser.add_argument("--results_dir", type=str, default="results/dropout_eval",
                        help="Directory containing evaluation results")
    parser.add_argument("--output_dir", type=str, default="results/figures",
                        help="Directory for output figures")
    parser.add_argument("--variant", type=str, default="both", choices=["1", "2", "both"],
                        help="Which variant to visualize")
    
    # Styling options
    parser.add_argument("--colormap", type=str, default="RdYlGn",
                        help="Matplotlib colormap for heatmaps")
    parser.add_argument("--style", type=str, default="paper",
                        choices=["paper", "poster", "presentation"],
                        help="Figure style preset")
    parser.add_argument("--figsize", type=str, default="10,8",
                        help="Figure size as 'width,height' in inches")
    parser.add_argument("--dpi", type=int, default=300,
                        help="DPI for saved figures")
    parser.add_argument("--format", type=str, default="png",
                        choices=["png", "pdf", "svg", "eps"],
                        help="Output format")
    
    # Analysis options
    parser.add_argument("--annotate", action="store_true", default=True,
                        help="Add value annotations to heatmap cells")
    parser.add_argument("--no_annotate", dest="annotate", action="store_false")
    parser.add_argument("--compare_policies", action="store_true", default=True,
                        help="Generate policy comparison plots")
    parser.add_argument("--generate_summary", action="store_true", default=True,
                        help="Generate summary statistics")
    
    return parser.parse_args()


# =============================================================================
# Style Configuration
# =============================================================================

STYLE_PRESETS = {
    "paper": {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "figure.dpi": 300,
    },
    "poster": {
        "font.family": "sans-serif",
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.titlesize": 24,
        "figure.dpi": 150,
    },
    "presentation": {
        "font.family": "sans-serif",
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 20,
        "figure.dpi": 150,
    },
}

# Policy display names and colors
POLICY_DISPLAY = {
    "M1_policy": {"name": "M1 (Vision+Proprio)", "color": "#3498db"},
    "M2_policy": {"name": "M2 (Vision+Proprio+Force)", "color": "#e74c3c"},
    "M3_mix": {"name": "M3 (Vision+Proprio, Dropout-trained)", "color": "#2ecc71"},
    "M4_mix": {"name": "M4 (Vision+Proprio+Force, Dropout-trained)", "color": "#9b59b6"},
}

# Phase display names
PHASE_DISPLAY = {
    "reach": "A: Reach",
    "grasp": "B: Grasp",
    "lift": "C: Lift",
    "transport": "C: Transport",
    "place": "D: Place",
}


def apply_style(style_name: str):
    """Apply matplotlib style preset."""
    plt.rcParams.update(STYLE_PRESETS.get(style_name, STYLE_PRESETS["paper"]))
    if HAS_SEABORN:
        sns.set_style("whitegrid")


# =============================================================================
# Data Loading
# =============================================================================

def load_summary(filepath: str) -> dict:
    """Load evaluation summary from JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_all_summaries(results_dir: str) -> Dict[str, List[dict]]:
    """Load all evaluation summaries from directory.
    
    Returns:
        Dict mapping (policy_name, variant) to summary data
    """
    summaries = {"variant1": {}, "variant2": {}}
    
    # Load individual summaries
    pattern = os.path.join(results_dir, "*_summary.json")
    for filepath in glob.glob(pattern):
        try:
            summary = load_summary(filepath)
            variant_key = f"variant{summary['variant']}"
            policy_name = summary['policy_name']
            summaries[variant_key][policy_name] = summary
        except Exception as e:
            print(f"[WARN] Failed to load {filepath}: {e}")
    
    # Also try combined summary
    combined_path = os.path.join(results_dir, "all_summaries.json")
    if os.path.exists(combined_path):
        try:
            with open(combined_path, 'r') as f:
                all_summaries = json.load(f)
            for summary in all_summaries:
                variant_key = f"variant{summary['variant']}"
                policy_name = summary['policy_name']
                summaries[variant_key][policy_name] = summary
        except Exception as e:
            print(f"[WARN] Failed to load combined summary: {e}")
    
    return summaries


# =============================================================================
# Visualization Functions
# =============================================================================

def create_heatmap(
    matrix: np.ndarray,
    x_labels: List[str],
    y_labels: List[str],
    title: str,
    xlabel: str,
    ylabel: str,
    colormap: str = "RdYlGn",
    annotate: bool = True,
    vmin: float = 0.0,
    vmax: float = 1.0,
    figsize: Tuple[float, float] = (10, 8),
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Create a single heatmap."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Create heatmap
    im = ax.imshow(matrix, cmap=colormap, aspect='auto', vmin=vmin, vmax=vmax)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Success Rate', rotation=270, labelpad=15)
    
    # Set ticks
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add annotations
    if annotate:
        for i in range(len(y_labels)):
            for j in range(len(x_labels)):
                value = matrix[i, j]
                # Choose text color based on background
                text_color = "white" if value < 0.5 else "black"
                ax.text(j, i, f"{value:.0%}", ha="center", va="center",
                        color=text_color, fontsize=8)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Add grid lines
    ax.set_xticks(np.arange(len(x_labels) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(y_labels) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    return fig, ax


def plot_single_policy_heatmap(
    summary: dict,
    output_path: str,
    colormap: str = "RdYlGn",
    annotate: bool = True,
    figsize: Tuple[float, float] = (10, 8),
    dpi: int = 300,
):
    """Generate heatmap for a single policy."""
    matrix = np.array(summary["success_matrix"])
    conditions = summary["conditions"]
    durations = summary["durations"]
    variant = summary["variant"]
    policy_name = summary["policy_name"]
    
    # Format labels
    if variant == 1:
        y_labels = [PHASE_DISPLAY.get(c, c) for c in conditions]
        ylabel = "Manipulation Phase"
        title = f"{policy_name}: Phase-Based Dropout Sensitivity"
    else:
        y_labels = [f"Step {c}" for c in conditions]
        ylabel = "Dropout Onset (steps)"
        title = f"{policy_name}: Time-Based Dropout Sensitivity"
    
    x_labels = [f"{d} steps\n({d*0.05:.1f}s)" for d in durations]
    xlabel = "Dropout Duration"
    
    fig, ax = create_heatmap(
        matrix=matrix,
        x_labels=x_labels,
        y_labels=y_labels,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        colormap=colormap,
        annotate=annotate,
        figsize=figsize,
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_policy_comparison(
    summaries: Dict[str, dict],
    output_path: str,
    colormap: str = "RdYlGn",
    annotate: bool = True,
    figsize: Tuple[float, float] = (16, 12),
    dpi: int = 300,
):
    """Generate comparison heatmaps for all policies."""
    n_policies = len(summaries)
    if n_policies == 0:
        print("[WARN] No summaries to compare")
        return
    
    # Determine grid layout
    if n_policies <= 2:
        nrows, ncols = 1, n_policies
    elif n_policies <= 4:
        nrows, ncols = 2, 2
    else:
        nrows = (n_policies + 2) // 3
        ncols = 3
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_policies == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Get variant info from first summary
    first_summary = list(summaries.values())[0]
    variant = first_summary["variant"]
    
    for idx, (policy_name, summary) in enumerate(summaries.items()):
        ax = axes[idx]
        matrix = np.array(summary["success_matrix"])
        conditions = summary["conditions"]
        durations = summary["durations"]
        
        # Format labels
        if variant == 1:
            y_labels = [PHASE_DISPLAY.get(c, c) for c in conditions]
            ylabel = "Phase"
        else:
            y_labels = [f"{c}" for c in conditions]
            ylabel = "Onset Step"
        
        x_labels = [str(d) for d in durations]
        xlabel = "Duration (steps)"
        
        # Create heatmap on axis
        im = ax.imshow(matrix, cmap=colormap, aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_yticklabels(y_labels, fontsize=8)
        
        if annotate and matrix.shape[0] * matrix.shape[1] <= 50:
            for i in range(len(y_labels)):
                for j in range(len(x_labels)):
                    value = matrix[i, j]
                    text_color = "white" if value < 0.5 else "black"
                    ax.text(j, i, f"{value:.0%}", ha="center", va="center",
                            color=text_color, fontsize=7)
        
        display_name = POLICY_DISPLAY.get(policy_name, {}).get("name", policy_name)
        ax.set_title(display_name, fontsize=10, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
    
    # Hide unused axes
    for idx in range(n_policies, len(axes)):
        axes[idx].set_visible(False)
    
    # Add shared colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Success Rate', rotation=270, labelpad=15)
    
    # Add title
    variant_name = "Phase-Based" if variant == 1 else "Time-Based"
    fig.suptitle(f"Policy Comparison: {variant_name} Dropout Sensitivity", 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_difference_map(
    summaries: Dict[str, dict],
    baseline_policy: str,
    comparison_policy: str,
    output_path: str,
    figsize: Tuple[float, float] = (10, 8),
    dpi: int = 300,
):
    """Generate difference heatmap between two policies."""
    if baseline_policy not in summaries or comparison_policy not in summaries:
        print(f"[WARN] Cannot compare: {baseline_policy} vs {comparison_policy}")
        return
    
    baseline = summaries[baseline_policy]
    comparison = summaries[comparison_policy]
    
    baseline_matrix = np.array(baseline["success_matrix"])
    comparison_matrix = np.array(comparison["success_matrix"])
    
    diff_matrix = comparison_matrix - baseline_matrix
    
    variant = baseline["variant"]
    conditions = baseline["conditions"]
    durations = baseline["durations"]
    
    if variant == 1:
        y_labels = [PHASE_DISPLAY.get(c, c) for c in conditions]
        ylabel = "Manipulation Phase"
    else:
        y_labels = [f"Step {c}" for c in conditions]
        ylabel = "Dropout Onset"
    
    x_labels = [str(d) for d in durations]
    xlabel = "Duration (steps)"
    
    baseline_name = POLICY_DISPLAY.get(baseline_policy, {}).get("name", baseline_policy)
    comparison_name = POLICY_DISPLAY.get(comparison_policy, {}).get("name", comparison_policy)
    title = f"Improvement: {comparison_name} vs {baseline_name}"
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use diverging colormap centered at 0
    max_abs = max(abs(diff_matrix.min()), abs(diff_matrix.max()))
    im = ax.imshow(diff_matrix, cmap="RdBu", aspect='auto', 
                   vmin=-max_abs, vmax=max_abs)
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Δ Success Rate', rotation=270, labelpad=15)
    
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    
    # Annotate
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            value = diff_matrix[i, j]
            text_color = "white" if abs(value) > max_abs * 0.5 else "black"
            sign = "+" if value > 0 else ""
            ax.text(j, i, f"{sign}{value:.0%}", ha="center", va="center",
                    color=text_color, fontsize=8)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_summary_line(
    summaries: Dict[str, dict],
    output_path: str,
    figsize: Tuple[float, float] = (12, 5),
    dpi: int = 300,
):
    """Generate line plot showing average success vs duration for each policy."""
    fig, ax = plt.subplots(figsize=figsize)
    
    for policy_name, summary in summaries.items():
        matrix = np.array(summary["success_matrix"])
        durations = summary["durations"]
        
        # Average across conditions
        avg_success = matrix.mean(axis=0)
        std_success = matrix.std(axis=0)
        
        display = POLICY_DISPLAY.get(policy_name, {"name": policy_name, "color": "gray"})
        
        ax.plot(durations, avg_success, 'o-', label=display["name"], 
                color=display["color"], linewidth=2, markersize=6)
        ax.fill_between(durations, avg_success - std_success, avg_success + std_success,
                        alpha=0.2, color=display["color"])
    
    ax.set_xlabel("Dropout Duration (steps)")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 1)
    ax.set_xlim(min(durations) - 5, max(durations) + 5)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    variant = list(summaries.values())[0]["variant"]
    variant_name = "Phase-Based" if variant == 1 else "Time-Based"
    ax.set_title(f"Average Success vs Duration ({variant_name} Dropout)")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_statistics_report(
    summaries: Dict[str, dict],
    output_path: str,
):
    """Generate text report with statistics."""
    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("DROPOUT SENSITIVITY ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        for policy_name, summary in summaries.items():
            matrix = np.array(summary["success_matrix"])
            conditions = summary["conditions"]
            durations = summary["durations"]
            variant = summary["variant"]
            
            display_name = POLICY_DISPLAY.get(policy_name, {}).get("name", policy_name)
            
            f.write(f"\n{'-'*40}\n")
            f.write(f"Policy: {display_name}\n")
            f.write(f"Variant: {'Phase-Based' if variant == 1 else 'Time-Based'}\n")
            f.write(f"{'-'*40}\n\n")
            
            # Overall statistics
            f.write("Overall Statistics:\n")
            f.write(f"  Mean Success Rate: {matrix.mean():.2%}\n")
            f.write(f"  Std Dev: {matrix.std():.2%}\n")
            f.write(f"  Min Success Rate: {matrix.min():.2%}\n")
            f.write(f"  Max Success Rate: {matrix.max():.2%}\n\n")
            
            # Per-condition statistics
            f.write("Per-Condition Analysis:\n")
            for i, cond in enumerate(conditions):
                cond_label = PHASE_DISPLAY.get(cond, cond) if variant == 1 else f"Onset {cond}"
                row = matrix[i]
                f.write(f"  {cond_label}: mean={row.mean():.2%}, min={row.min():.2%}, max={row.max():.2%}\n")
            
            f.write("\n")
            
            # Duration sensitivity (L50 approximation)
            f.write("Duration Sensitivity:\n")
            clean_success = matrix[:, 0].mean()  # Shortest duration as proxy
            for i, dur in enumerate(durations):
                avg_success = matrix[:, i].mean()
                relative = avg_success / clean_success if clean_success > 0 else 0
                f.write(f"  Duration {dur} steps: {avg_success:.2%} ({relative:.1%} of baseline)\n")
            
            f.write("\n")
        
        # Comparison section
        if len(summaries) > 1:
            f.write("\n" + "=" * 60 + "\n")
            f.write("POLICY COMPARISON\n")
            f.write("=" * 60 + "\n\n")
            
            policy_names = list(summaries.keys())
            matrices = {p: np.array(summaries[p]["success_matrix"]) for p in policy_names}
            
            # Overall ranking
            rankings = [(p, matrices[p].mean()) for p in policy_names]
            rankings.sort(key=lambda x: x[1], reverse=True)
            
            f.write("Overall Ranking (by mean success rate):\n")
            for i, (policy, score) in enumerate(rankings):
                display = POLICY_DISPLAY.get(policy, {}).get("name", policy)
                f.write(f"  {i+1}. {display}: {score:.2%}\n")
            
            f.write("\n")
            
            # Pairwise improvements
            if "M1_policy" in summaries:
                f.write("Improvement over M1 (baseline):\n")
                baseline = matrices.get("M1_policy")
                for policy in policy_names:
                    if policy != "M1_policy" and policy in matrices:
                        diff = matrices[policy] - baseline
                        display = POLICY_DISPLAY.get(policy, {}).get("name", policy)
                        f.write(f"  {display}: avg improvement = {diff.mean():+.2%}\n")
    
    print(f"Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    
    # Apply style
    apply_style(args.style)
    
    # Parse figsize
    figsize = tuple(float(x) for x in args.figsize.split(","))
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load summaries
    print(f"\nLoading results from: {args.results_dir}")
    all_summaries = load_all_summaries(args.results_dir)
    
    # Process Variant 1 (Phase-based)
    if args.variant in ["1", "both"] and all_summaries["variant1"]:
        print(f"\n{'='*40}")
        print("Generating Variant 1 (Phase-Based) visualizations...")
        print(f"{'='*40}")
        
        variant1 = all_summaries["variant1"]
        
        # Individual heatmaps
        for policy_name, summary in variant1.items():
            output_path = os.path.join(args.output_dir, f"{policy_name}_variant1_heatmap.{args.format}")
            plot_single_policy_heatmap(
                summary, output_path,
                colormap=args.colormap,
                annotate=args.annotate,
                figsize=figsize,
                dpi=args.dpi,
            )
        
        # Comparison plot
        if args.compare_policies and len(variant1) > 1:
            output_path = os.path.join(args.output_dir, f"comparison_variant1.{args.format}")
            plot_policy_comparison(
                variant1, output_path,
                colormap=args.colormap,
                annotate=args.annotate,
                figsize=(16, 12),
                dpi=args.dpi,
            )
        
        # Difference maps
        if "M1_policy" in variant1:
            for policy in ["M2_policy", "M3_mix", "M4_mix"]:
                if policy in variant1:
                    output_path = os.path.join(args.output_dir, f"diff_{policy}_vs_M1_variant1.{args.format}")
                    plot_difference_map(variant1, "M1_policy", policy, output_path, figsize, args.dpi)
        
        # Line plot
        output_path = os.path.join(args.output_dir, f"summary_line_variant1.{args.format}")
        plot_summary_line(variant1, output_path, figsize=(12, 5), dpi=args.dpi)
        
        # Statistics report
        if args.generate_summary:
            output_path = os.path.join(args.output_dir, "statistics_variant1.txt")
            generate_statistics_report(variant1, output_path)
    
    # Process Variant 2 (Time-based)
    if args.variant in ["2", "both"] and all_summaries["variant2"]:
        print(f"\n{'='*40}")
        print("Generating Variant 2 (Time-Based) visualizations...")
        print(f"{'='*40}")
        
        variant2 = all_summaries["variant2"]
        
        # Individual heatmaps
        for policy_name, summary in variant2.items():
            output_path = os.path.join(args.output_dir, f"{policy_name}_variant2_heatmap.{args.format}")
            plot_single_policy_heatmap(
                summary, output_path,
                colormap=args.colormap,
                annotate=args.annotate,
                figsize=figsize,
                dpi=args.dpi,
            )
        
        # Comparison plot
        if args.compare_policies and len(variant2) > 1:
            output_path = os.path.join(args.output_dir, f"comparison_variant2.{args.format}")
            plot_policy_comparison(
                variant2, output_path,
                colormap=args.colormap,
                annotate=args.annotate,
                figsize=(16, 12),
                dpi=args.dpi,
            )
        
        # Difference maps
        if "M1_policy" in variant2:
            for policy in ["M2_policy", "M3_mix", "M4_mix"]:
                if policy in variant2:
                    output_path = os.path.join(args.output_dir, f"diff_{policy}_vs_M1_variant2.{args.format}")
                    plot_difference_map(variant2, "M1_policy", policy, output_path, figsize, args.dpi)
        
        # Line plot
        output_path = os.path.join(args.output_dir, f"summary_line_variant2.{args.format}")
        plot_summary_line(variant2, output_path, figsize=(12, 5), dpi=args.dpi)
        
        # Statistics report
        if args.generate_summary:
            output_path = os.path.join(args.output_dir, "statistics_variant2.txt")
            generate_statistics_report(variant2, output_path)
    
    print(f"\n{'='*60}")
    print(f"All visualizations saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

