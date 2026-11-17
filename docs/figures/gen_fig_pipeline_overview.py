#!/usr/bin/env python3
"""Generate publication-ready IEEE TCSVT / TIP overview pipeline figure for T-SCOPE v3.

Exact v3 alignment (zero v1/v2 drift, zero arrow/text collisions):
  - Panel 1: 16:9 video I_t (640x384), 6-DOF Cauchy-IRLS affine MC (H/8->H/4, <0.75 px Static
    Identity Snap) + soft sensor-noise deadzone coring (0.012), and Multi-Codec SeverityConditioner
    (num_families=2: 0=libx265 HEVC, 1=libx264 AVC; 6-D polynomial-hinge basis phi(s), 2-layer GELU MLP).
  - Panel 2: TemporalSCOPE (6,108,814 params): FiLM Multi-Scale U-Net (H..H/8 ctx_fuse) + motion-warped
    H/4 Bottleneck ConvGRU, Smoothstep STE A_eff,t + anti-flicker BG attention lock, bounded FG residual
    (|Delta_obj| <= 0.025*g(s)) + motion-locked BG EMA, AND IWA Long-GOP P-Frame Bypass
    (forward_inter_bypass_step at H/2: -96.7% GMACs on T=32).
  - Panel 3: Halo-Guarded Non-Negative QG Rate Allocation: 3x3 max-pool FG halo guard -> native 16x16 px
    QG pooling (0.50*Avg + 0.50*Max, 40x24 grid at 640x384) -> strictly non-negative DeltaQP in [0, +7.14]
    above tau=0.38 (DeltaQP_fg = 0.0) + fg_safe P-carry -> RLE addroi (qoffset in [0.00, +0.14]).
  - Panel 4: Standard-Compliant Multi-Codec yuv420p Encoding (libx265 & libx264, GOP=32 LDP/RA, PTS-sorted
    ffprobe packet bytes) + Differentiable Multi-Codec 8x8 DCT Intra/Inter Proxy Backward STE.
  - Panel 5: Single-Teacher Training Objective (Frozen Faster R-CNN R50-FPN v2 ONLY) vs. Zero-Shot
    Evaluation (YOLOv8s, RetinaNet, FCOS, SSDLite, Swin-T + ByteTrack MOTA/IDF1/HOTA across MOT17/MOT20/KITTI).
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def create_pipeline_figure(output_png: Path, output_pdf: Path) -> None:
    fig, ax = plt.subplots(figsize=(18.4, 9.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["mathtext.fontset"] = "dejavusans"

    c_in_bg, c_in_bd, c_in_tx = "#EFF6FF", "#2563EB", "#1E3A8A"
    c_cd_bg, c_cd_bd, c_cd_tx = "#F5F3FF", "#7C3AED", "#4C1D95"
    c_nt_bg, c_nt_bd, c_nt_tx = "#F0FDF4", "#16A34A", "#14532D"
    c_bp_bg, c_bp_bd, c_bp_tx = "#ECFEFF", "#0891B2", "#164E63"
    c_ed_bg, c_ed_bd, c_ed_tx = "#FEF3C7", "#D97706", "#78350F"
    c_qp_bg, c_qp_bd, c_qp_tx = "#FFEDD5", "#EA580C", "#7C2D12"
    c_co_bg, c_co_bd, c_co_tx = "#F1F5F9", "#475569", "#0F172A"
    c_px_bg, c_px_bd, c_px_tx = "#FDF2F8", "#DB2777", "#831843"
    c_tr_bg, c_tr_bd, c_tr_tx = "#EEF2FF", "#4F46E5", "#1E1B4B"

    def draw_container(
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        subtitle: str = "",
        bg_col: str = "#FFFFFF",
        border_col: str = "#CBD5E1",
        header_col: str | None = None,
        title_fs: float = 8.8,
    ) -> None:
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.22,rounding_size=0.95",
            facecolor=bg_col,
            edgecolor=border_col,
            linewidth=1.35,
            zorder=1,
        )
        ax.add_patch(box)
        if header_col:
            hbar = FancyBboxPatch(
                (x + 0.4, y + h - 3.9),
                w - 0.8,
                3.4,
                boxstyle="round,pad=0.14,rounding_size=0.55",
                facecolor=header_col,
                edgecolor="none",
                zorder=2,
            )
            ax.add_patch(hbar)
            ax.text(
                x + w / 2,
                y + h - 2.2,
                title,
                ha="center",
                va="center",
                fontsize=9.4,
                fontweight="bold",
                color="#FFFFFF",
                zorder=3,
            )
        else:
            ax.text(
                x + w / 2,
                y + h - 1.9,
                title,
                ha="center",
                va="center",
                fontsize=title_fs,
                fontweight="bold",
                color="#1E293B",
                zorder=3,
            )
        if subtitle:
            ax.text(
                x + w / 2,
                y + h - 5.2,
                subtitle,
                ha="center",
                va="center",
                fontsize=7.4,
                fontstyle="italic",
                color="#475569",
                zorder=3,
            )

    def draw_module(
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        body: str = "",
        bg_col: str = "#FFFFFF",
        border_col: str = "#94A3B8",
        text_col: str = "#0F172A",
        title_fs: float = 8.1,
        body_fs: float = 6.8,
        ls: str = "-",
        zorder: int = 4,
    ) -> None:
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.18,rounding_size=0.65",
            facecolor=bg_col,
            edgecolor=border_col,
            linewidth=1.15,
            linestyle=ls,
            zorder=zorder,
        )
        ax.add_patch(box)
        if body:
            title_lines = title.count("\n") + 1
            top_pad = 1.4 if title_lines == 1 else 1.2
            title_block_h = 2.4 * title_lines
            ax.text(
                x + w / 2,
                y + h - top_pad,
                title,
                ha="center",
                va="top",
                fontsize=title_fs,
                fontweight="bold",
                color=text_col,
                linespacing=1.15,
                zorder=zorder + 1,
            )
            body_center_y = y + (h - title_block_h - 0.6) * 0.50
            ax.text(
                x + w / 2,
                body_center_y,
                body,
                ha="center",
                va="center",
                fontsize=body_fs,
                color=text_col,
                linespacing=1.20,
                zorder=zorder + 1,
            )
        else:
            ax.text(
                x + w / 2,
                y + h / 2,
                title,
                ha="center",
                va="center",
                fontsize=title_fs,
                fontweight="bold",
                color=text_col,
                zorder=zorder + 1,
            )

    def draw_arrow(
        p1: tuple[float, float],
        p2: tuple[float, float],
        col: str = "#1E293B",
        style: str = "-|>",
        lw: float = 1.4,
        ls: str = "-",
        rad: float = 0.0,
        label: str = "",
        label_pos: float = 0.5,
        label_offset: tuple[float, float] = (0.0, 1.0),
        label_col: str | None = None,
        fontsize: float = 7.1,
    ) -> None:
        conn = f"arc3,rad={rad}" if rad != 0.0 else "arc3"
        arrow = FancyArrowPatch(
            p1,
            p2,
            arrowstyle=style,
            color=col,
            linewidth=lw,
            linestyle=ls,
            connectionstyle=conn,
            mutation_scale=10.0,
            zorder=8,
        )
        ax.add_patch(arrow)
        if label:
            lx = p1[0] + (p2[0] - p1[0]) * label_pos + label_offset[0]
            ly = p1[1] + (p2[1] - p1[1]) * label_pos + label_offset[1]
            ax.text(
                lx,
                ly,
                label,
                ha="center",
                va="center",
                fontsize=fontsize,
                fontweight="bold",
                color=label_col or col,
                zorder=9,
                bbox=dict(boxstyle="round,pad=0.14", facecolor="#FFFFFF", edgecolor="none", alpha=0.95),
            )

    def draw_polyline_arrow(
        pts: list[tuple[float, float]],
        col: str = "#1E293B",
        lw: float = 1.5,
        ls: str = "-",
        label: str = "",
        label_xy: tuple[float, float] | None = None,
        fontsize: float = 7.2,
    ) -> None:
        xs = [p[0] for p in pts[:-1]] + [pts[-1][0]]
        ys = [p[1] for p in pts[:-1]] + [pts[-1][1]]
        ax.plot(xs[:-1], ys[:-1], color=col, linewidth=lw, linestyle=ls, solid_capstyle="round", zorder=8)
        arr = FancyArrowPatch(
            pts[-2],
            pts[-1],
            arrowstyle="-|>",
            color=col,
            linewidth=lw,
            linestyle=ls,
            mutation_scale=10.5,
            zorder=8,
        )
        ax.add_patch(arr)
        if label and label_xy is not None:
            ax.text(
                label_xy[0],
                label_xy[1],
                label,
                ha="center",
                va="center",
                fontsize=fontsize,
                fontweight="bold",
                color=col,
                zorder=9,
                bbox=dict(boxstyle="round,pad=0.16", facecolor="#FFFFFF", edgecolor=col, linewidth=0.6, alpha=0.96),
            )

    # ==========================================
    # 0. HEADER
    # ==========================================
    ax.text(
        50.0,
        98.0,
        "T-SCOPE v3: Motion-Compensated Semantic Pre-Editing & Halo-Guarded QG Rate Allocation for VCM",
        ha="center",
        va="center",
        fontsize=13.0,
        fontweight="bold",
        color="#0F172A",
    )
    ax.text(
        50.0,
        95.6,
        "Standard-Compliant Multi-Codec 4:2:0 Video Coding (libx265 HEVC & libx264 AVC, GOP=32 LDP/RA) with Single-Teacher Training & Zero-Shot VCM Transfer",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#475569",
    )

    # Note: y in [8.2, 93.4] leaves clean bottom routing corridor y in [4.6, 7.6] for cross-panel buses!

    # ==========================================
    # 1. PANEL 1: INPUTS, 6-DOF IRLS MC & CONDITIONER
    # ==========================================
    draw_container(
        0.8,
        8.2,
        15.4,
        85.2,
        "1. Align & Condition",
        "6-DOF Affine MC & Multi-Codec FiLM",
        bg_col="#F8FAFC",
        border_col="#94A3B8",
        header_col="#1E40AF",
    )

    draw_module(
        1.8,
        76.8,
        13.4,
        9.8,
        r"Video Sequence $\{I_t\}_{t=1}^T$",
        "16:9 Aspect-Preserving RGB\n" + r"$3 \times 384 \times 640$ ($T=32$ GOP)",
        bg_col=c_in_bg,
        border_col=c_in_bd,
        text_col=c_in_tx,
    )

    draw_module(
        1.8,
        59.6,
        13.4,
        14.6,
        "6-DOF Cauchy-IRLS MC",
        r"Coarse $H/8$ + IRLS ($H/8 \to H/4$)" + "\n"
        + r"$<0.75$ px Static Identity Snap" + "\n"
        + r"Soft Coring ($\tau_{\mathrm{core}}=0.012$):" + "\n"
        + r"$\Delta I_t^{\mathrm{MC}} = \mathrm{core}((I_t - \tilde{I}_{t-1})\odot V_t)$",
        bg_col=c_in_bg,
        border_col=c_in_bd,
        text_col=c_in_tx,
        body_fs=6.6,
    )

    draw_module(
        2.4,
        49.8,
        12.2,
        7.2,
        r"Concat $[I_t, \Delta I_t^{\mathrm{MC}}]$",
        r"$6 \times H \times W$ Aligned Input",
        bg_col="#DBEAFE",
        border_col="#3B82F6",
        text_col="#1E40AF",
        title_fs=8.0,
    )

    draw_arrow((8.5, 76.8), (8.5, 74.2), col="#2563EB", lw=1.3)
    draw_arrow((8.5, 59.6), (8.5, 57.0), col="#2563EB", lw=1.3)

    draw_module(
        1.8,
        31.2,
        13.4,
        15.2,
        "Codec & Rate Target",
        r"Family $f \in \{0:\mathrm{x265}, 1:\mathrm{x264}\}$" + "\n"
        + r"Embed $e_f \in \mathbb{R}^8$ (Jointly Trained)" + "\n"
        + r"$\mathrm{CRF} \in \{27, 32, 37, 42\}$" + "\n"
        + r"$s = \mathrm{clip}((q - 22)/25, 0, 1)$",
        bg_col=c_cd_bg,
        border_col=c_cd_bd,
        text_col=c_cd_tx,
        body_fs=6.6,
    )

    draw_module(
        1.8,
        9.8,
        13.4,
        18.4,
        "SeverityConditioner",
        r"6-D Poly-Hinge Basis $\phi(s)$:" + "\n"
        + r"$[s, s^2, s^3, h_{.50}, h_{.75}, h_{.90}]$" + "\n"
        + "2-Layer Linear + GELU MLP\n"
        + r"$\Downarrow$" + "\n"
        + r"FiLM Vector $c \in \mathbb{R}^{64}$",
        bg_col="#EDE9FE",
        border_col="#6D28D9",
        text_col="#4C1D95",
        body_fs=6.7,
    )
    draw_arrow((8.5, 31.2), (8.5, 28.2), col="#7C3AED", lw=1.3)

    # ==========================================
    # 2. PANEL 2: TEMPORALSCOPE ARCHITECTURE (6.11M)
    # ==========================================
    draw_container(
        17.1,
        8.2,
        32.8,
        85.2,
        "2. TemporalSCOPE Pre-Editor (6,108,814 Params)",
        "Multi-Scale FiLM U-Net + H/4 ConvGRU & Long-GOP Inter-Frame Bypass",
        bg_col="#F8FAFC",
        border_col="#94A3B8",
        header_col="#15803D",
    )

    # Upper sub-container
    draw_container(
        18.1,
        46.6,
        30.8,
        40.2,
        "Recurrent Multi-Scale Backbone & IWA P-Bypass",
        "",
        bg_col="#FFFFFF",
        border_col="#CBD5E1",
        title_fs=8.3,
    )

    draw_module(
        19.0,
        59.6,
        9.2,
        21.2,
        "FiLM Encoder\n(H -> H/8)",
        r"enc1 ($H$, no GN)" + "\n"
        + r"enc2 ($H/2$, GN)" + "\n"
        + r"bottleneck ($H/4$)" + "\n"
        + r"+ enc3 ($H/8$) $\to$" + "\n"
        + r"ctx_fuse ($192$ ch)",
        bg_col=c_nt_bg,
        border_col=c_nt_bd,
        text_col=c_nt_tx,
        title_fs=7.7,
        body_fs=6.5,
    )

    draw_module(
        29.2,
        59.6,
        9.4,
        21.2,
        "Motion-Warped\nH/4 ConvGRU",
        r"$\tilde{H}_{t-1} = \mathcal{W}_{\theta_t}(H_{t-1})\odot V_{H/4}$" + "\n"
        + r"$H_t = \mathrm{GRU}(F_{3,\mathrm{ctx}}, \tilde{H}_{t-1})$" + "\n\n"
        + "Stabilizes Moving &\nStatic Camera State",
        bg_col="#DCFCE7",
        border_col="#16A34A",
        text_col="#14532D",
        title_fs=7.7,
        body_fs=6.4,
    )

    draw_module(
        39.6,
        59.6,
        8.4,
        21.2,
        "FiLM Decoder\n& 5 Heads",
        r"dec2 ($H/2$) $\to$ dec1 ($H$)" + "\n"
        + "Skip Fuse (f2, f1)\n"
        + "Outputs:\n"
        + r"$A_t, \Delta_{\mathrm{obj}}, \Delta_{\mathrm{bg}}$" + "\n"
        + r"BlurMix, $g_{\mathrm{ema}}$",
        bg_col=c_nt_bg,
        border_col=c_nt_bd,
        text_col=c_nt_tx,
        title_fs=7.7,
        body_fs=6.5,
    )

    # Skip connection cleanly inside gap [80.8, 84.4] below subcontainer header (84.9)!
    draw_arrow(
        (23.6, 80.9),
        (43.8, 80.9),
        col="#15803D",
        lw=1.15,
        ls=":",
        rad=-0.10,
        label="Multi-Scale Skips (f1, f2)",
        label_pos=0.5,
        label_offset=(0.0, 1.35),
        fontsize=6.7,
    )

    # Fast IWA P-frame bypass bar inside backbone box (2-line compact text, zero overflow)
    draw_module(
        19.0,
        47.8,
        29.0,
        10.2,
        "IWA Inter-Frame Bypass (t mod 32 != 0): -96.7% GMACs (137.8 -> 4.52/f)",
        r"Skips enc1..dec1 & ConvGRU; warps $(A_{t-1}, A^{\mathrm{eff}}_{t-1}, I_{\mathrm{bg},t-1})$"
        + "\n"
        + r"+ lightweight $H/2$ inter_refine (1,604 params) + disocclusion gate",
        bg_col=c_bp_bg,
        border_col=c_bp_bd,
        text_col=c_bp_tx,
        title_fs=7.15,
        body_fs=6.35,
        ls="--",
    )

    draw_arrow((28.2, 70.2), (29.2, 70.2), col="#16A34A", lw=1.3)
    draw_arrow((38.6, 70.2), (39.6, 70.2), col="#16A34A", lw=1.3)
    draw_arrow((14.6, 53.4), (19.0, 69.5), col="#2563EB", lw=1.4, label=r"$6\times H\times W$", label_pos=0.46)
    draw_arrow((15.2, 19.0), (19.0, 63.0), col="#7C3AED", lw=1.25, ls="--", rad=0.14, label=r"FiLM $c$", label_pos=0.40)

    # Lower sub-container
    draw_container(
        18.1,
        9.6,
        30.8,
        34.8,
        "Smoothstep STE & Dual-Branch Pixel Synthesis",
        "",
        bg_col="#FFFFFF",
        border_col="#CBD5E1",
        title_fs=8.3,
    )

    draw_module(
        19.0,
        23.0,
        14.1,
        17.6,
        "Foreground Task Branch",
        r"Smoothstep STE $A_t^{\mathrm{eff}} \in [0,1]$" + "\n"
        + r"+ Anti-Flicker BG Lock ($w_{\mathrm{lock}}$)" + "\n"
        + r"$|\Delta_{\mathrm{obj},t}| \leq 0.025\,(0.3 + 0.7s)$" + "\n"
        + r"$I_{\mathrm{obj},t} = \mathrm{clip}(I_t + \Delta_{\mathrm{obj},t}, 0, 1)$",
        bg_col=c_ed_bg,
        border_col=c_ed_bd,
        text_col=c_ed_tx,
        title_fs=7.7,
        body_fs=6.45,
    )

    draw_module(
        33.9,
        23.0,
        14.1,
        17.6,
        "Motion-Locked BG EMA",
        r"Adaptive $5\times 5$ / Multi-Scale Blur" + "\n"
        + r"Static Conf $C_{\mathrm{bg},t} = G_5(V_t e^{-28 m_t})$" + "\n"
        + r"Damped $\Delta_{\mathrm{bg},t}$, $w_{\mathrm{temp}} \leq 0.92$:" + "\n"
        + r"$I_{\mathrm{bg},t} = w_{\mathrm{temp}}\tilde{I}_{\mathrm{bg},t-1} + (1-w_{\mathrm{temp}})I_{\mathrm{bg}}^{\mathrm{sp}}$",
        bg_col=c_ed_bg,
        border_col=c_ed_bd,
        text_col=c_ed_tx,
        title_fs=7.7,
        body_fs=6.35,
    )

    draw_module(
        19.6,
        10.8,
        27.8,
        9.8,
        r"Asymmetric Synthesis:  $I_{\mathrm{edit},t} = A_t^{\mathrm{eff}} \odot I_{\mathrm{obj},t} + (1 - A_t^{\mathrm{eff}}) \odot I_{\mathrm{bg},t}$",
        "Zero Foreground Blur Leakage + Temporally Locked P/B-Frame Background Residuals",
        bg_col="#FDE68A",
        border_col="#D97706",
        text_col="#78350F",
        title_fs=7.55,
        body_fs=6.55,
    )

    draw_polyline_arrow([(47.2, 59.6), (48.45, 59.6), (48.45, 45.5), (26.0, 45.5), (26.0, 40.6)], col="#D97706", lw=1.2)
    draw_polyline_arrow([(48.45, 45.5), (41.0, 45.5), (41.0, 40.6)], col="#D97706", lw=1.2)
    draw_arrow((26.0, 23.0), (28.5, 20.6), col="#D97706", lw=1.2)
    draw_arrow((41.0, 23.0), (38.5, 20.6), col="#D97706", lw=1.2)

    # ==========================================
    # 3. PANEL 3: HALO-GUARDED NON-NEGATIVE QG dQP
    # ==========================================
    draw_container(
        50.8,
        8.2,
        16.4,
        85.2,
        "3. Halo-Guarded QG dQP",
        "Native 16x16 px Grid (40x24)",
        bg_col="#F8FAFC",
        border_col="#94A3B8",
        header_col="#C2410C",
    )

    draw_module(
        51.8,
        69.2,
        14.4,
        17.4,
        r"$3\times 3$ FG Halo Guard & Pool",
        r"$A_t^{\mathrm{guard}} = \mathrm{MaxPool}_{3\times 3}(A_t^{\mathrm{eff}})$" + "\n"
        + r"Native $16\times 16$ px QGs ($40\times 24$):" + "\n"
        + r"$A_t^{\mathrm{QG}} = 0.50\,\mathrm{Avg}_{16} + 0.50\,\mathrm{Max}_{16}$" + "\n"
        + "(Any FG px >= 0.76 zeroes +dQP)",
        bg_col=c_qp_bg,
        border_col=c_qp_bd,
        text_col=c_qp_tx,
        title_fs=7.7,
        body_fs=6.45,
    )

    draw_module(
        51.8,
        39.2,
        14.4,
        26.6,
        "Non-Negative QG Coarsening",
        r"Threshold $\tau = 0.38$, $\kappa_{\max} = 4.0$:" + "\n\n"
        + r"$\bullet$ Foreground + Halo ($A_t^{\mathrm{QG}} \geq \tau$):" + "\n"
        + r"$\mathbf{\Delta QP}_{\mathrm{fg}} = \mathbf{0.0}$  (Unmodified)" + "\n\n"
        + r"$\bullet$ Background ($A_t^{\mathrm{QG}} < \tau$):" + "\n"
        + r"$\mathbf{\Delta QP}_{\mathrm{bg}} \in [0.0, +7.14]$" + "\n"
        + r"+ fg_safe Monotonic P-Carry",
        bg_col="#FFF7ED",
        border_col="#EA580C",
        text_col="#9A3412",
        title_fs=7.7,
        body_fs=6.45,
    )

    draw_module(
        51.8,
        9.8,
        14.4,
        26.0,
        "Full-Grid addroi & Script",
        "Zero 16x16 Macro-Cell Cap\n"
        + "Emitted Offset Range:\n"
        + r"$q_{\mathrm{offset}} = \Delta\mathrm{QP}/51 \in [0.00, +0.14]$" + "\n\n"
        + "Horizontal Run-Length Merge\n"
        + "+ Auto Filter-Script Offload\n"
        + "(-filter_script:v / -/vf >4k ch)",
        bg_col=c_qp_bg,
        border_col=c_qp_bd,
        text_col=c_qp_tx,
        title_fs=7.7,
        body_fs=6.45,
    )

    draw_arrow((59.0, 69.2), (59.0, 65.8), col="#EA580C", lw=1.3)
    draw_arrow((59.0, 39.2), (59.0, 35.8), col="#EA580C", lw=1.3)
    # Route A_eff cleanly from FG Task Branch via gap x=50.35 up into Panel 3 top box!
    draw_polyline_arrow(
        [(33.1, 35.5), (50.35, 35.5), (50.35, 78.0), (51.8, 78.0)],
        col="#EA580C",
        lw=1.35,
        label=r"$A_t^{\mathrm{eff}}$",
        label_xy=(50.35, 56.5),
    )

    # ==========================================
    # 4. PANEL 4: MULTI-CODEC yuv420p & STE PROXY
    # ==========================================
    draw_container(
        68.1,
        8.2,
        15.2,
        85.2,
        "4. Multi-Codec & STE",
        "Real 4:2:0 Forward / Proxy Bwd",
        bg_col="#F8FAFC",
        border_col="#94A3B8",
        header_col="#334155",
    )

    draw_module(
        69.1,
        51.0,
        13.2,
        35.6,
        "Standard 4:2:0 Video Codecs\n(Forward Pass)",
        "f=0: libx265 (HEVC Main)\n"
        + "f=1: libx264 (AVC High/Main)\n"
        + "-pix_fmt yuv420p (Strict)\n"
        + "preset=medium, aq-mode=2\n"
        + "keyint=32: LDP (BF=0) &\n"
        + "Random Access (IBBP, BF=4)\n\n"
        + "PTS-Sorted ffprobe Packets\n"
        + r"$\Rightarrow \hat{I}_t^{\mathrm{real}},\; R_{\mathrm{tot}}, R_I, R_P$",
        bg_col=c_co_bg,
        border_col=c_co_bd,
        text_col=c_co_tx,
        title_fs=7.6,
        body_fs=6.45,
    )

    draw_module(
        69.1,
        9.8,
        13.2,
        36.2,
        "Differentiable Multi-Codec\nProxy (Backward STE)",
        r"$8\times 8$ DCT + $16\times 16$ Inter MC" + "\n"
        + "Codec-Family Scaling (f=0,1)\n"
        + "Full QG Nearest 8x8 Mapping\n"
        + r"$C^\infty$ Deadzone / SKIP Rate" + "\n\n"
        + "Straight-Through Coupling:\n"
        + r"$\hat{I}_t = \hat{I}_t^{\mathrm{px}} + \mathrm{sg}(\hat{I}_t^{\mathrm{real}} - \hat{I}_t^{\mathrm{px}})$" + "\n"
        + r"$R_t = R_t^{\mathrm{px}} + \mathrm{sg}(R_t^{\mathrm{real}} - R_t^{\mathrm{px}})$",
        bg_col=c_px_bg,
        border_col=c_px_bd,
        text_col=c_px_tx,
        title_fs=7.6,
        body_fs=6.35,
    )

    # Clean orthogonal routing in dedicated bottom corridor (zero crossings through Panel 3 boxes!)
    draw_polyline_arrow(
        [(33.5, 10.8), (33.5, 6.7), (67.6, 6.7), (67.6, 68.0), (69.1, 68.0)],
        col="#2563EB",
        lw=1.45,
        label=r"Pre-Edited Video $I_{\mathrm{edit},t}$",
        label_xy=(50.5, 6.7),
    )
    draw_polyline_arrow(
        [(66.2, 22.8), (68.55, 22.8), (68.55, 60.0), (69.1, 60.0)],
        col="#EA580C",
        lw=1.35,
        label="addroi",
        label_xy=(68.55, 42.0),
    )
    draw_arrow((75.7, 51.0), (75.7, 46.0), col="#475569", lw=1.25, style="<->", label="STE", label_pos=0.5, label_offset=(2.3, 0.0))
    draw_polyline_arrow(
        [(75.7, 9.8), (75.7, 5.0), (26.0, 5.0), (26.0, 10.8)],
        col="#DB2777",
        lw=1.4,
        ls="--",
        label=r"Backward Proxy STE Gradients $\nabla \mathcal{L}_{\mathrm{total}}$",
        label_xy=(50.5, 5.0),
    )

    # ==========================================
    # 5. PANEL 5: SINGLE-TEACHER TRAIN & ZERO-SHOT EVAL
    # ==========================================
    draw_container(
        84.2,
        8.2,
        15.0,
        85.2,
        "5. Train vs. Zero-Shot",
        "De-Confounded VCM Evaluation",
        bg_col="#F8FAFC",
        border_col="#94A3B8",
        header_col="#3730A3",
    )

    draw_module(
        85.2,
        57.6,
        13.0,
        29.0,
        "Single-Teacher Training\n(Train Split ONLY)",
        "Frozen Faster R-CNN v2\n"
        + r"FPN $\{P_2, P_3, P_4, P_5\}$ ONLY" + "\n"
        + "(use_yolov8s=False)\n\n"
        + r"$\mathcal{L}_{\mathrm{task}}$ (FPN + FG Pix/Grad)" + "\n"
        + r"$+ \mathcal{L}_{\mathrm{attn}}$ (Tversky + Sat)" + "\n"
        + r"$+ \mathcal{L}_{\mathrm{rate}}(R_I + 4.5 R_P + \log)$" + "\n"
        + r"$+ \mathcal{L}_{\mathrm{temp}} + \mathcal{L}_{\mathrm{dqp\_fg/mono}}$",
        bg_col=c_tr_bg,
        border_col=c_tr_bd,
        text_col=c_tr_tx,
        title_fs=7.6,
        body_fs=6.35,
    )

    draw_module(
        85.2,
        9.8,
        13.0,
        44.8,
        "Zero-Shot VCM Evaluation\n(Held-Out Test)",
        "4 Unseen Detectors + ViT:\n"
        + r"$\bullet$ YOLOv8s (CSPDarknet)" + "\n"
        + r"$\bullet$ RetinaNet R50-FPN v2" + "\n"
        + r"$\bullet$ FCOS R50-FPN" + "\n"
        + r"$\bullet$ SSDLite MobileNetV3" + "\n"
        + r"$\bullet$ Swin-T Feature SNR" + "\n\n"
        + "Tracking & Multi-Dataset:\n"
        + r"$\bullet$ ByteTrack: MOTA/IDF1/HOTA" + "\n"
        + r"$\bullet$ MOT17, MOT20, KITTI" + "\n\n"
        + "Statistical Rigor:\n"
        + "PCHIP BDBR + 10k Paired\n"
        + "Clip-Bootstrap 95% CIs",
        bg_col="#F5F3FF",
        border_col="#6366F1",
        text_col="#312E81",
        title_fs=7.6,
        body_fs=6.3,
    )

    draw_arrow((82.3, 72.0), (85.2, 72.0), col="#1E293B", lw=1.3, label=r"$\hat{I}_t$", label_pos=0.5)
    draw_polyline_arrow(
        [(82.3, 58.0), (83.75, 58.0), (83.75, 34.0), (85.2, 34.0)],
        col="#1E293B",
        lw=1.3,
        label=r"$\hat{I}_t, R_t$",
        label_xy=(83.75, 47.0),
    )
    draw_polyline_arrow(
        [(85.2, 62.5), (83.35, 62.5), (83.35, 28.0), (82.3, 28.0)],
        col="#DB2777",
        lw=1.25,
        ls="--",
    )

    # ==========================================
    # 6. BOTTOM LEGEND BAR
    # ==========================================
    legend_box = FancyBboxPatch(
        (6.0, 0.7),
        88.0,
        2.9,
        boxstyle="round,pad=0.14,rounding_size=0.55",
        facecolor="#FFFFFF",
        edgecolor="#CBD5E1",
        linewidth=1.1,
        zorder=7,
    )
    ax.add_patch(legend_box)

    items = [
        (8.0, 2.15, "#2563EB", "-", "Aligned Video & Tensor Flow"),
        (28.0, 2.15, "#0891B2", "--", "IWA Inter-Frame P-Bypass (-96.7% GMACs)"),
        (53.0, 2.15, "#EA580C", "-", r"Halo-Guarded Non-Negative QG $\Delta\mathrm{QP}$"),
        (75.5, 2.15, "#DB2777", "--", "Backward Proxy STE Gradients"),
    ]
    for lx, ly, lcol, lstyle, ltext in items:
        arr = FancyArrowPatch(
            (lx, ly),
            (lx + 2.4, ly),
            arrowstyle="-|>",
            color=lcol,
            linewidth=1.6,
            linestyle=lstyle,
            mutation_scale=8.5,
            zorder=8,
        )
        ax.add_patch(arr)
        ax.text(lx + 2.9, ly, ltext, va="center", fontsize=7.6, fontweight="bold", color="#334155", zorder=8)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {output_png} and {output_pdf}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent.parent
    out_dir = repo_root / "docs" / "figures"
    create_pipeline_figure(out_dir / "fig_pipeline_overview.png", out_dir / "fig_pipeline_overview.pdf")
