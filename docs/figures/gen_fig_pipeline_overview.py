#!/usr/bin/env python3
"""Generate publication-ready pipeline overview figure for T-SCOPE v3.

Visual design inspired by clean VCM architecture conventions (cf. Zhao et al., arXiv:2512.15331
Figs. 1-2):
  - Soft neutral pill-rounded containers, minimal text density (module names & math symbols only,
    zero prose paragraphs inside boxes), and iconic visual schematics.
  - Subfigure (a): Overall Hybrid-STE Training & Standard-Compliant Testing Framework
    (perspective-stacked real MOT17 video frames -> T-SCOPE pre-editor icon -> coupled Real
    4:2:0 Codec + Differentiable Proxy STE -> Vision Task Analyzer & loss supervision).
  - Subfigure (b): Two-Branch Spatio-Temporal Structure of T-SCOPE
    (3D isometric green cuboids for 6-DOF IRLS + ConvGRU + BG EMA temporal branch, 3D isometric
    terracotta cuboids for FiLM U-Net spatial branch, IWA P-frame bypass, Smoothstep blend, and
    3x3 halo-guarded 40x24 QG Delta-QP map).
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Circle, Rectangle
from PIL import Image


def _load_sample_frames(fig_dir: Path) -> list[np.ndarray]:
    """Load real MOT17-11 frames from mot17_real_thumbs.jpg; deterministic fallback otherwise."""
    strip_path = fig_dir / "mot17_real_thumbs.jpg"
    if strip_path.is_file():
        strip = Image.open(strip_path).convert("RGB")
        tw = strip.width // 5
        th = strip.height
        if tw >= 64 and th >= 48:
            return [
                np.asarray(strip.crop((i * tw, 0, (i + 1) * tw, th)), dtype=np.float32) / 255.0
                for i in range(5)
            ]

    rng = np.random.default_rng(2026)
    frames = []
    yy, xx = np.mgrid[0:144, 0:240]
    for k in range(5):
        sky = np.stack(
            [
                0.72 - 0.18 * (yy / 144.0),
                0.78 - 0.16 * (yy / 144.0),
                0.86 - 0.12 * (yy / 144.0),
            ],
            axis=-1,
        )
        sky[yy > 84] = [0.36, 0.38, 0.40]
        for px, col in [(60 + k * 7, [0.82, 0.25, 0.22]), (140 - k * 5, [0.20, 0.48, 0.78])]:
            ped = (np.abs(xx - px) < 10) & (yy > 56) & (yy < 120)
            sky[ped] = col
        sky = np.clip(sky + rng.normal(0, 0.012, size=sky.shape), 0.0, 1.0)
        frames.append(sky.astype(np.float32))
    return frames


def _shear_image_to_axes(
    ax: plt.Axes,
    img: np.ndarray,
    x0: float,
    y0: float,
    w: float,
    h: float,
    shear_y: float = 2.8,
    zorder: int = 5,
    border_col: str = "#1F2937",
    lw: float = 1.15,
) -> None:
    """Render an RGB image onto a perspective-sheared parallelogram patch in axes coordinates."""
    gh, gw = 96, 150
    pil_im = Image.fromarray((np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)).resize(
        (gw, gh), Image.Resampling.LANCZOS
    )
    arr = np.asarray(pil_im, dtype=np.float32) / 255.0

    u = np.linspace(0.0, 1.0, gw + 1)
    v = np.linspace(0.0, 1.0, gh + 1)
    uu, vv = np.meshgrid(u, v)
    X = x0 + uu * w
    Y = (y0 + (1.0 - vv) * h) + uu * shear_y
    ax.pcolormesh(X, Y, arr, shading="flat", zorder=zorder, rasterized=True)

    outline = Polygon(
        [
            (x0, y0),
            (x0 + w, y0 + shear_y),
            (x0 + w, y0 + h + shear_y),
            (x0, y0 + h),
        ],
        closed=True,
        facecolor="none",
        edgecolor=border_col,
        linewidth=lw,
        joinstyle="round",
        zorder=zorder + 0.2,
    )
    ax.add_patch(outline)


def _draw_cuboid(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    dx: float,
    dy: float,
    front_col: str,
    top_col: str,
    side_col: str,
    edge_col: str = "#333333",
    lw: float = 0.95,
    zorder: int = 5,
) -> None:
    """Draw a clean 3D isometric tensor cuboid (front, top, right faces)."""
    front = Polygon(
        [(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
        closed=True,
        facecolor=front_col,
        edgecolor=edge_col,
        linewidth=lw,
        joinstyle="round",
        zorder=zorder,
    )
    top = Polygon(
        [(x, y + h), (x + dx, y + h + dy), (x + w + dx, y + h + dy), (x + w, y + h)],
        closed=True,
        facecolor=top_col,
        edgecolor=edge_col,
        linewidth=lw,
        joinstyle="round",
        zorder=zorder,
    )
    side = Polygon(
        [(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)],
        closed=True,
        facecolor=side_col,
        edgecolor=edge_col,
        linewidth=lw,
        joinstyle="round",
        zorder=zorder,
    )
    ax.add_patch(front)
    ax.add_patch(top)
    ax.add_patch(side)


def _draw_pill_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    fc: str,
    ec: str,
    radius: float = 1.0,
    lw: float = 1.25,
    ls: str = "-",
    zorder: int = 3,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle=ls,
        zorder=zorder,
    )
    ax.add_patch(patch)


def _arrow(
    ax: plt.Axes,
    p1: tuple[float, float],
    p2: tuple[float, float],
    col: str,
    lw: float = 1.75,
    ls: str = "-",
    style: str = "-|>",
    rad: float = 0.0,
    ms: float = 11.0,
    zorder: int = 7,
) -> None:
    conn = f"arc3,rad={rad}" if rad != 0.0 else "arc3"
    arr = FancyArrowPatch(
        p1,
        p2,
        arrowstyle=style,
        color=col,
        linewidth=lw,
        linestyle=ls,
        connectionstyle=conn,
        mutation_scale=ms,
        zorder=zorder,
    )
    ax.add_patch(arr)


def _poly_arrow(
    ax: plt.Axes,
    pts: list[tuple[float, float]],
    col: str,
    lw: float = 1.75,
    ls: str = "-",
    ms: float = 11.0,
    zorder: int = 7,
) -> None:
    if len(pts) > 2:
        ax.plot(
            [p[0] for p in pts[:-1]],
            [p[1] for p in pts[:-1]],
            color=col,
            linewidth=lw,
            linestyle=ls,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=zorder,
        )
    _arrow(ax, pts[-2], pts[-1], col=col, lw=lw, ls=ls, ms=ms, zorder=zorder)


def _draw_dqp_heatmap_icon(ax: plt.Axes, x: float, y: float, w: float, h: float, zorder: int = 6) -> None:
    """Draw a miniature 40x24 QG Delta-QP grid icon (cool blue=0 FG/halo, warm amber=+dQP BG)."""
    ny, nx = 8, 12
    grid = np.ones((ny, nx), dtype=np.float32) * 0.82
    for cy, cx in [(4, 3), (4, 8)]:
        for dy in range(-2, 3):
            for dx in range(-1, 2):
                yy, xx = cy + dy, cx + dx
                if 0 <= yy < ny and 0 <= xx < nx:
                    dist = max(abs(dy) - 1, abs(dx))
                    grid[yy, xx] = 0.05 if dist <= 0 else min(grid[yy, xx], 0.28)
    cmap = plt.get_cmap("YlOrBr")
    rgb = cmap(grid)[..., :3]
    rgb[grid < 0.15] = [0.22, 0.49, 0.78]
    rgb[(grid >= 0.15) & (grid < 0.35)] = [0.70, 0.84, 0.94]

    cw, ch = w / nx, h / ny
    for r in range(ny):
        for c in range(nx):
            rect = Rectangle(
                (x + c * cw, y + (ny - 1 - r) * ch),
                cw,
                ch,
                facecolor=rgb[r, c],
                edgecolor="#FFFFFF",
                linewidth=0.35,
                zorder=zorder,
            )
            ax.add_patch(rect)
    border = Rectangle((x, y), w, h, facecolor="none", edgecolor="#334155", linewidth=0.95, zorder=zorder + 0.2)
    ax.add_patch(border)


def create_pipeline_figure(output_png: Path, output_pdf: Path) -> None:
    frames = _load_sample_frames(output_png.resolve().parent)

    fig, ax = plt.subplots(figsize=(14.4, 8.1), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("auto")
    ax.axis("off")

    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["mathtext.fontset"] = "stixsans"

    CANVAS_BG = "#EEEEEE"
    CANVAS_BD = "#8A8A8A"
    COL_TEST = "#447834"      # Forest green: testing / standard deployment flow
    COL_TRAIN = "#8F1D1D"     # Deep maroon: training / STE supervision flow
    COL_IWA = "#0D8A82"       # Teal: IWA inter-frame bypass
    COL_DQP = "#C2591B"       # Burnt-orange: QG Delta-QP addroi control
    CARD_BG = "#D6E2F0"       # Soft steel-blue module card
    CARD_BD = "#3B5E8C"       # Crisp navy-blue border
    TASK_BG = "#5882C1"       # Solid steel-blue task analyzer
    TASK_BD = "#2B4C7E"

    # =========================================================================
    # PANEL (a): OVERALL FRAMEWORK (Top Pill Container)
    # =========================================================================
    _draw_pill_box(ax, 2.0, 54.0, 96.0, 44.2, fc=CANVAS_BG, ec=CANVAS_BD, radius=3.2, lw=1.7, zorder=1)
    ax.text(
        50.0,
        50.8,
        "(a) Overall Hybrid Real-Codec STE Training & Standard-Compliant Deployment Pipeline",
        ha="center",
        va="center",
        fontsize=10.1,
        fontweight="bold",
        color="#1F2937",
    )

    # 1. Input video stacked real MOT17 frames
    stack_offsets = [(0.0, 0.0), (2.7, 1.3), (5.4, 2.6)]
    for idx, (dx, dy) in enumerate(stack_offsets):
        _shear_image_to_axes(
            ax,
            frames[idx],
            x0=5.0 + dx,
            y0=69.5 - dy * 0.45,
            w=7.6,
            h=14.4,
            shear_y=3.0,
            zorder=4 + idx,
        )
    ax.text(11.6, 63.8, r"Input video $\{I_t\}$", ha="center", va="center", fontsize=9.2, fontweight="bold", color="#1F2937")
    ax.text(11.6, 60.7, r"640×384 (32-f GOP)", ha="center", va="center", fontsize=7.6, color="#4B5563")

    # Legend neatly stacked in bottom-left of Panel (a)
    _arrow(ax, (20.2, 63.4), (24.8, 63.4), col=COL_TEST, lw=1.85, ms=9.5)
    ax.text(25.4, 63.4, "testing (deployment)", ha="left", va="center", fontsize=7.9, color="#1F2937")
    _arrow(ax, (20.2, 60.0), (24.8, 60.0), col=COL_TRAIN, lw=1.85, ms=9.5)
    ax.text(25.4, 60.0, "training (Hybrid STE)", ha="left", va="center", fontsize=7.9, color="#1F2937")
    _arrow(ax, (20.2, 56.6), (24.8, 56.6), col=COL_IWA, lw=1.65, ls="--", ms=9.5)
    ax.text(25.4, 56.6, "IWA P-bypass (113.2 FPS)", ha="left", va="center", fontsize=7.9, color="#1F2937")

    # Input -> Pre-editor dual arrows
    _arrow(ax, (19.0, 80.2), (26.2, 80.2), col=COL_TEST, lw=1.9)
    _arrow(ax, (19.0, 75.2), (26.2, 75.2), col=COL_TRAIN, lw=1.9)

    # 2. T-SCOPE Pre-Editor Card
    _draw_pill_box(ax, 26.6, 67.6, 17.8, 21.6, fc=CARD_BG, ec=CARD_BD, radius=1.5, lw=1.4, zorder=3)
    ax.text(35.5, 91.5, "T-SCOPE pre-editor", ha="center", va="center", fontsize=9.3, fontweight="bold", color="#1F2937")

    # Iconic U-Net trapezoids + ConvGRU bottleneck inside Pre-Editor card
    enc_trap = Polygon(
        [(28.4, 71.4), (32.1, 73.7), (32.1, 81.3), (28.4, 83.6)],
        closed=True,
        facecolor="#335C94",
        edgecolor="#1E3A5F",
        linewidth=1.0,
        zorder=5,
    )
    dec_trap = Polygon(
        [(36.1, 73.7), (39.8, 71.4), (39.8, 83.6), (36.1, 81.3)],
        closed=True,
        facecolor="#335C94",
        edgecolor="#1E3A5F",
        linewidth=1.0,
        zorder=5,
    )
    ax.add_patch(enc_trap)
    ax.add_patch(dec_trap)
    ax.plot([32.3, 35.9], [79.4, 79.4], color="#1E3A5F", lw=1.2, ls="--", zorder=5)
    ax.plot([32.3, 35.9], [75.6, 75.6], color="#1E3A5F", lw=1.2, ls="--", zorder=5)
    ax.text(34.1, 77.5, "GRU", ha="center", va="center", fontsize=7.0, fontweight="bold", color="#1E3A5F", zorder=6)

    # Small FiLM (f, CRF) + IRLS badge at bottom of Pre-Editor
    _draw_pill_box(ax, 28.6, 68.5, 11.2, 2.2, fc="#EEF4FB", ec="#5C7FA8", radius=0.5, lw=0.9, zorder=5)
    ax.text(34.2, 69.6, r"FiLM $(f, \mathrm{CRF})$ + IRLS", ha="center", va="center", fontsize=6.8, fontweight="bold", color="#1E3A5F", zorder=6)

    # Teal dashed IWA bypass arc cleanly inside upper space of Pre-Editor card
    _arrow(ax, (28.6, 84.3), (39.8, 84.3), col=COL_IWA, lw=1.55, ls="--", rad=-0.26, ms=9.0, zorder=6)
    ax.text(
        34.2,
        87.2,
        "IWA P-bypass",
        ha="center",
        va="center",
        fontsize=6.9,
        fontweight="bold",
        color=COL_IWA,
        bbox=dict(boxstyle="round,pad=0.10", facecolor=CARD_BG, edgecolor="none"),
        zorder=7,
    )

    # Mini QG Delta-QP heatmap badge inside right side of Pre-Editor
    _draw_dqp_heatmap_icon(ax, 41.0, 74.2, 2.5, 4.2, zorder=6)
    ax.text(42.25, 72.5, r"$\Delta\mathrm{QP}$", ha="center", va="center", fontsize=7.1, fontweight="bold", color=COL_DQP, zorder=6)

    # 3. Real Codec (Top) & Differentiable Virtual Proxy (Bottom)
    _draw_pill_box(ax, 54.0, 81.4, 16.4, 13.0, fc=CARD_BG, ec=CARD_BD, radius=1.3, lw=1.35, zorder=3)
    ax.text(50.6, 91.0, "real\ncodec", ha="center", va="center", fontsize=8.3, fontweight="bold", color="#1F2937")
    ax.text(62.2, 92.7, "x265 / x264 (4:2:0)", ha="center", va="center", fontsize=7.5, fontweight="bold", color="#1E3A5F", zorder=5)

    # Iconic frame -> bitstream filmstrip -> decoded frame
    _draw_pill_box(ax, 55.6, 83.4, 3.0, 7.2, fc="#4A74B0", ec="#233D63", radius=0.3, lw=0.9, zorder=5)
    _draw_pill_box(ax, 60.8, 83.2, 2.6, 7.6, fc="#1F2937", ec="#111827", radius=0.25, lw=0.9, zorder=5)
    for sy in [84.2, 86.0, 87.8, 89.4]:
        ax.add_patch(Rectangle((61.2, sy), 0.55, 0.9, facecolor="#F9FAFB", edgecolor="none", zorder=6))
        ax.add_patch(Rectangle((62.45, sy), 0.55, 0.9, facecolor="#F9FAFB", edgecolor="none", zorder=6))
    _draw_pill_box(ax, 65.6, 83.4, 3.0, 7.2, fc="#7498CA", ec="#233D63", radius=0.3, lw=0.9, zorder=5)
    _arrow(ax, (58.8, 87.0), (60.6, 87.0), col="#1E3A5F", lw=1.3, ms=8, zorder=6)
    _arrow(ax, (63.6, 87.0), (65.4, 87.0), col="#1E3A5F", lw=1.3, ms=8, zorder=6)

    # Virtual Codec / Differentiable Proxy Card
    _draw_pill_box(ax, 54.0, 63.6, 16.4, 12.6, fc=CARD_BG, ec=CARD_BD, radius=1.3, lw=1.35, zorder=3)
    ax.text(50.5, 72.2, "codec\nproxy", ha="center", va="center", fontsize=8.3, fontweight="bold", color="#1F2937")
    ax.text(62.2, 74.5, "8×8 DCT + 16×16 MC", ha="center", va="center", fontsize=7.4, fontweight="bold", color="#1E3A5F", zorder=5)

    _draw_pill_box(ax, 55.6, 65.4, 3.0, 6.8, fc="#4A74B0", ec="#233D63", radius=0.3, lw=0.9, zorder=5)
    qx = np.linspace(59.8, 64.4, 40)
    qy = 68.8 + 2.3 * np.exp(-((qx - 62.1) / 1.05) ** 2)
    ax.plot([59.6, 64.6], [67.0, 67.0], color="#334155", lw=0.9, zorder=5)
    ax.plot(qx, qy, color="#255494", lw=1.6, zorder=6)
    _draw_pill_box(ax, 65.6, 65.4, 3.0, 6.8, fc="#7498CA", ec="#233D63", radius=0.3, lw=0.9, zorder=5)
    _arrow(ax, (58.7, 68.8), (59.9, 68.8), col="#1E3A5F", lw=1.2, ms=7.5, zorder=6)
    _arrow(ax, (64.4, 68.8), (65.5, 68.8), col="#1E3A5F", lw=1.2, ms=7.5, zorder=6)

    # Pre-editor -> Codecs routing
    _poly_arrow(ax, [(44.4, 82.0), (47.2, 82.0), (47.2, 87.8), (53.8, 87.8)], col=COL_TEST, lw=1.9)
    _poly_arrow(ax, [(43.6, 76.3), (48.8, 76.3), (48.8, 84.4), (53.8, 84.4)], col=COL_DQP, lw=1.5, ls="--", ms=9.5)
    ax.text(51.3, 83.1, "addroi", ha="center", va="center", fontsize=6.9, fontweight="bold", color=COL_DQP)
    _poly_arrow(ax, [(44.4, 72.0), (47.2, 72.0), (47.2, 68.8), (53.8, 68.8)], col=COL_TRAIN, lw=1.9)

    # Vertical STE bridge between Real Codec (forward) and Proxy (backward)
    _arrow(ax, (62.2, 81.2), (62.2, 76.4), col=COL_TRAIN, lw=1.55, ls="--", style="<|-|>", ms=9.0, zorder=7)
    ax.text(
        65.2,
        78.8,
        "STE",
        ha="center",
        va="center",
        fontsize=7.6,
        fontweight="bold",
        color=COL_TRAIN,
        bbox=dict(boxstyle="round,pad=0.12", facecolor=CANVAS_BG, edgecolor="none"),
        zorder=8,
    )

    # Downward Loss Arrows from Proxy
    _arrow(ax, (58.4, 63.5), (58.4, 59.0), col=COL_TRAIN, lw=1.8, ms=10)
    ax.text(58.4, 57.1, r"$\mathcal{L}_{\mathrm{bg}}, \mathcal{L}_{\mathrm{temp}}$", ha="center", va="center", fontsize=8.4, fontweight="bold", color="#111827")
    _arrow(ax, (66.4, 63.5), (66.4, 59.0), col=COL_TRAIN, lw=1.8, ms=10)
    ax.text(66.4, 57.1, r"$\mathcal{L}_{\mathrm{rate}}$", ha="center", va="center", fontsize=8.4, fontweight="bold", color="#111827")

    # 4. Vision Task Analyzer (Right solid blue card)
    _draw_pill_box(ax, 76.8, 71.4, 18.2, 13.8, fc=TASK_BG, ec=TASK_BD, radius=1.6, lw=1.45, zorder=4)
    ax.text(85.9, 80.4, "Vision task\nanalyzer", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#FFFFFF", zorder=6)
    ax.text(
        85.9,
        74.2,
        "Train: Frozen FRCNN\nTest: 5 Det. + ByteTrack",
        ha="center",
        va="center",
        fontsize=7.3,
        color="#EEF4FF",
        zorder=6,
    )

    _poly_arrow(ax, [(70.4, 87.8), (73.8, 87.8), (73.8, 80.4), (76.6, 80.4)], col=COL_TEST, lw=1.9)
    _poly_arrow(ax, [(70.4, 68.8), (73.8, 68.8), (73.8, 75.8), (76.6, 75.8)], col=COL_TRAIN, lw=1.9)

    _arrow(ax, (85.9, 71.3), (85.9, 64.2), col=COL_TRAIN, lw=1.9, ms=10.5)
    ax.text(85.9, 62.0, r"$\mathcal{L}_{\mathrm{task}}$", ha="center", va="center", fontsize=8.9, fontweight="bold", color="#111827")

    # =========================================================================
    # PANEL (b): T-SCOPE TWO-BRANCH STRUCTURE (Bottom Pill Container, Zhao Fig. 2 style)
    # =========================================================================
    _draw_pill_box(ax, 2.0, 5.0, 96.0, 42.6, fc=CANVAS_BG, ec=CANVAS_BD, radius=3.2, lw=1.7, zorder=1)
    ax.text(
        50.0,
        1.8,
        "(b) Two-Branch Spatio-Temporal Structure of T-SCOPE Pre-Editor & Halo-Guarded QG Rate Controller",
        ha="center",
        va="center",
        fontsize=10.1,
        fontweight="bold",
        color="#1F2937",
    )

    TEMP_ARROW = "#4C7C38"
    SPAT_ARROW = "#984B22"
    TEMP_F, TEMP_T, TEMP_S = "#9DC183", "#B9D6A3", "#769B5E"
    SPAT_F, SPAT_T, SPAT_S = "#CD8858", "#E0A67E", "#A16238"

    # Top-right legend in Panel (b)
    _arrow(ax, (75.2, 44.6), (79.7, 44.6), col=TEMP_ARROW, lw=1.8, ms=9.5)
    ax.text(80.3, 44.6, "temporal / MC branch", ha="left", va="center", fontsize=7.9, color="#1F2937")
    _arrow(ax, (75.2, 41.6), (79.7, 41.6), col=SPAT_ARROW, lw=1.8, ms=9.5)
    ax.text(80.3, 41.6, "spatial FiLM branch", ha="left", va="center", fontsize=7.9, color="#1F2937")

    # Left Input Cuboid Stack [I_{t-1}, I_t]
    ax.text(9.8, 42.6, r"$[I_{t-1}, \dots, I_t]$", ha="center", va="center", fontsize=8.7, fontweight="bold", color="#1F2937")
    _draw_cuboid(ax, 5.2, 24.4, 2.0, 11.8, 1.8, 2.1, "#4B79B4", "#729CD0", "#345988", zorder=4)
    _draw_cuboid(ax, 8.0, 23.4, 2.0, 11.8, 1.8, 2.1, SPAT_F, SPAT_T, SPAT_S, zorder=5)
    _draw_cuboid(ax, 10.8, 22.4, 2.0, 11.8, 1.8, 2.1, "#4B79B4", "#729CD0", "#345988", zorder=6)
    ax.text(7.8, 19.8, r"$H\times W\times 3$", ha="center", va="center", fontsize=7.5, color="#374151")

    # Clean bottom identity skip line into bottom of ⊕ (zero crossing over HxWx3 text!)
    _poly_arrow(ax, [(12.2, 22.3), (12.2, 7.8), (75.2, 7.8), (75.2, 17.5)], col="#374151", lw=1.35, ms=9.5, zorder=5)
    ax.text(43.8, 9.4, r"Clean foreground skip $I_t$ (Smoothstep STE protection)", ha="center", va="center", fontsize=7.5, color="#374151")

    # --- UPPER BRANCH: Temporal / Motion-Compensated Branch ---
    _poly_arrow(ax, [(14.6, 33.4), (17.6, 33.4), (17.6, 35.6), (20.2, 35.6)], col=TEMP_ARROW, lw=1.8)

    _draw_pill_box(ax, 20.4, 32.8, 8.8, 5.6, fc="#DCEBBE", ec="#567D3E", radius=0.8, lw=1.15, zorder=5)
    ax.text(24.8, 35.6, "6-DOF IRLS\n+ coring", ha="center", va="center", fontsize=7.5, fontweight="bold", color="#1F3A14", zorder=6)

    _arrow(ax, (29.3, 35.6), (32.4, 35.6), col=TEMP_ARROW, lw=1.8)
    _draw_cuboid(ax, 32.7, 31.2, 2.7, 8.8, 1.6, 1.9, TEMP_F, TEMP_T, TEMP_S, zorder=5)
    ax.text(34.6, 29.3, r"$\Delta I_{\mathrm{MC},t}$", ha="center", va="center", fontsize=7.9, fontweight="bold", color="#1F2937")

    _arrow(ax, (37.2, 35.6), (40.8, 35.6), col=TEMP_ARROW, lw=1.8)
    _draw_pill_box(ax, 41.0, 32.8, 9.2, 5.6, fc="#DCEBBE", ec="#567D3E", radius=0.8, lw=1.15, zorder=5)
    ax.text(45.6, 35.6, "Warped\nH/4 ConvGRU", ha="center", va="center", fontsize=7.5, fontweight="bold", color="#1F3A14", zorder=6)

    _arrow(ax, (50.3, 35.6), (53.4, 35.6), col=TEMP_ARROW, lw=1.8)
    _draw_cuboid(ax, 53.7, 32.0, 2.5, 7.2, 1.4, 1.7, TEMP_F, TEMP_T, TEMP_S, zorder=5)
    ax.text(55.5, 29.8, r"$H/4\times W/4$", ha="center", va="center", fontsize=7.3, color="#374151")

    _arrow(ax, (57.8, 35.6), (61.2, 35.6), col=TEMP_ARROW, lw=1.8)
    _draw_pill_box(ax, 61.4, 32.8, 9.2, 5.6, fc="#DCEBBE", ec="#567D3E", radius=0.8, lw=1.15, zorder=5)
    ax.text(66.0, 35.6, "Locked\nBG EMA", ha="center", va="center", fontsize=7.5, fontweight="bold", color="#1F3A14", zorder=6)

    # Dashed IWA shortcut sailing cleanly above upper cuboids (zero text collisions!)
    _arrow(ax, (24.8, 38.6), (66.0, 38.6), col=COL_IWA, lw=1.55, ls="--", rad=-0.18, ms=9.5, zorder=7)
    ax.text(
        45.4,
        45.2,
        r"IWA P-bypass ($-96.7\%$ GMACs)",
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        color=COL_IWA,
        bbox=dict(boxstyle="round,pad=0.14", facecolor=CANVAS_BG, edgecolor="none"),
        zorder=8,
    )

    # --- LOWER BRANCH: Spatial FiLM Multi-Scale U-Net Branch ---
    _poly_arrow(ax, [(14.6, 26.2), (17.6, 26.2), (17.6, 19.6), (20.2, 19.6)], col=SPAT_ARROW, lw=1.8)

    _draw_pill_box(ax, 20.4, 16.8, 8.8, 5.6, fc="#F5DFC9", ec="#984B22", radius=0.8, lw=1.15, zorder=5)
    ax.text(24.8, 19.6, "FiLM cond\n" + r"$(f, s)$", ha="center", va="center", fontsize=7.5, fontweight="bold", color="#5C280D", zorder=6)

    _arrow(ax, (29.3, 19.6), (32.4, 19.6), col=SPAT_ARROW, lw=1.8)
    _draw_cuboid(ax, 32.7, 14.5, 2.2, 10.0, 1.5, 1.8, SPAT_F, SPAT_T, SPAT_S, zorder=5)
    _draw_cuboid(ax, 36.8, 15.9, 2.6, 7.4, 1.4, 1.6, SPAT_F, SPAT_T, SPAT_S, zorder=6)
    ax.text(36.2, 12.6, r"FiLM U-Net ($6.11\mathrm{M}$)", ha="center", va="center", fontsize=7.4, color="#374151")

    # Bidirectional fusion between Warped ConvGRU and U-Net bottleneck
    _arrow(ax, (45.6, 32.6), (45.6, 22.8), col="#334155", lw=1.45, ls="--", style="<|-|>", ms=8.5, zorder=6)

    _arrow(ax, (41.0, 19.6), (43.6, 19.6), col=SPAT_ARROW, lw=1.8)
    _draw_pill_box(ax, 43.8, 16.8, 7.6, 5.6, fc="#F5DFC9", ec="#984B22", radius=0.8, lw=1.15, zorder=5)
    ax.text(47.6, 19.6, "ResBlocks\n+ Decoder", ha="center", va="center", fontsize=7.4, fontweight="bold", color="#5C280D", zorder=6)

    _arrow(ax, (51.5, 19.6), (54.0, 19.6), col=SPAT_ARROW, lw=1.8)
    _draw_cuboid(ax, 54.3, 14.5, 2.4, 10.0, 1.5, 1.8, SPAT_F, SPAT_T, SPAT_S, zorder=5)
    ax.text(56.2, 12.6, r"5 Heads ($A_t^{\mathrm{eff}}$)", ha="center", va="center", fontsize=7.4, color="#374151")

    _arrow(ax, (58.4, 19.6), (61.2, 19.6), col=SPAT_ARROW, lw=1.8)
    _draw_pill_box(ax, 61.4, 16.8, 9.2, 5.6, fc="#F5DFC9", ec="#984B22", radius=0.8, lw=1.15, zorder=5)
    ax.text(66.0, 19.6, "Smoothstep\n+ 3×3 Halo", ha="center", va="center", fontsize=7.4, fontweight="bold", color="#5C280D", zorder=6)

    # --- SYMMETRIC MERGE NODE (Circle ⊕ at y=19.6) & DUAL OUTPUTS ---
    merge_c = Circle((75.2, 19.6), radius=1.95, facecolor="#FFFFFF", edgecolor="#1F2937", linewidth=1.5, zorder=7)
    ax.add_patch(merge_c)
    ax.text(75.2, 19.6, r"$\oplus$", ha="center", va="center", fontsize=12.5, fontweight="bold", color="#1F2937", zorder=8)

    # Top (BG EMA) + Left (Smoothstep FG) -> ⊕
    _poly_arrow(ax, [(70.6, 35.6), (75.2, 35.6), (75.2, 21.8)], col=TEMP_ARROW, lw=1.8)
    _arrow(ax, (70.6, 19.6), (73.1, 19.6), col=SPAT_ARROW, lw=1.8)

    # Output 1 (Upper-Right of Panel b): Real Pre-Edited Video Frame I_{edit,t}
    _poly_arrow(ax, [(77.2, 19.6), (80.6, 19.6), (80.6, 29.4), (83.6, 29.4)], col="#1F2937", lw=1.85)
    _shear_image_to_axes(ax, frames[0], x0=84.2, y0=23.4, w=7.2, h=12.4, shear_y=2.6, zorder=6)
    ax.text(88.4, 20.8, r"Edited $I_{\mathrm{edit},t}$", ha="center", va="center", fontsize=8.4, fontweight="bold", color="#1F2937")

    # Output 2 (Lower-Right of Panel b): 40x24 QG Delta-QP Map -> addroi
    _poly_arrow(ax, [(66.0, 16.8), (66.0, 13.6), (83.8, 13.6)], col=COL_DQP, lw=1.7, ls="--")
    _draw_dqp_heatmap_icon(ax, 84.3, 10.3, 7.4, 6.6, zorder=6)
    ax.text(88.0, 8.2, r"40×24 QG $\Delta\mathrm{QP}\in[0,+7.14]$", ha="center", va="center", fontsize=7.8, fontweight="bold", color=COL_DQP)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight", pad_inches=0.06, facecolor="#FFFFFF")
    fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.06, facecolor="#FFFFFF")
    plt.close(fig)
    print(f"Saved Zhao-style pipeline overview to {output_png} and {output_pdf}")


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent
    create_pipeline_figure(
        output_png=out_dir / "fig_pipeline_overview.png",
        output_pdf=out_dir / "fig_pipeline_overview.pdf",
    )
