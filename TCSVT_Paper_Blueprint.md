# IEEE TCSVT Q1 Blueprint: Motion-Compensated Semantic Pre-Editing and Halo-Guarded QG Rate Allocation for Standard-Compliant Video Coding for Machines (`tscope-v3`)

## 1. Title, Target Venue, and Calibrated Scope

- **Primary Title:** *T-SCOPE: Motion-Compensated Semantic Pre-Editing and Halo-Guarded Quantization-Group Rate Allocation for Standard-Compliant Video Coding for Machines*
- **Target Venue:** **IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)** (`\documentclass[journal]{IEEEtran}`, 12–13 pages). Direct video-domain successor to standard-compliant ICM pre-processors by Lu *et al.* (IEEE TCSVT, Dec. 2024) and Yang *et al.* (IEEE TCSVT, Jul. 2024), extending Dani *et al.* (`SCOPE`, 2026), Zhao *et al.* (IEEE DCC 2024 / `arXiv:2512.15331`), and Fernández-Menduiña *et al.* (`WA-RDO` / `IWA-RDO`, IEEE ICASSP 2026) to long-GOP inter-frame video coding.
- **Explicit Scope Calibration (`D3` / `F10` / `F11` / `F24`):**
  1. **Active Multi-Codec Scope:** `SeverityConditioner(num_families=2)` is jointly trained and evaluated across **`libx265` (HEVC Main 4:2:0, `family_id=0`)** and **`libx264` (AVC High/Main 4:2:0, `family_id=1`)** under explicit `-pix_fmt yuv420p` encoding (`F10`, `F13`). Extension to `VVenC` (H.266/VVC) and native 4K UHD bitstreams is explicitly deferred to future work.
  2. **Single-Teacher Training vs. Zero-Shot Generalization (`F04`):** All `tscope-v3` checkpoints are trained strictly with a **single frozen detector teacher** (`Faster R-CNN ResNet50-FPN v2`, `use_yolov8s=False`), reserving `YOLOv8s`, `RetinaNet ResNet50-FPN v2`, `FCOS ResNet50-FPN`, `SSDLite MobileNetV3-Large`, and `Swin-T` feature-space SNR—as well as `ByteTrack` multi-object tracking (`MOTA`, `IDF1`, `HOTA`) and secondary datasets (`MOT20`, multi-class `KITTI`)—for strictly unseen zero-shot evaluation.
  3. **Theoretical Motivation vs. Formal Proof:** References to the Data Processing Inequality (DPI) and task-oriented Rate-Accuracy bounds (Bajić, IEEE MIPR 2025) serve strictly as architectural design motivation rather than formal mutual-information proofs.
  4. **Codec Runtime Honesty (`F24`):** We make **no unmeasured claim** that `libx265` encodes faster after pre-editing; CPU codec wall-clock encoding/decoding latency is profiled empirically across unedited anchors, static keyframe `addroi`, and dynamic per-frame filter-script `addroi` (`eval_sota_and_tracking.py`).

---

## 2. Problem Formulation & Three Literature Bottlenecks

Let $\mathcal{I} = \{I_0, I_1, \dots, I_{T-1}\}$ with $I_t \in [0, 1]^{3 \times H \times W}$ denote an uncompressed widescreen video sequence ($H = 384, W = 640$, preserving native $16:9$ aspect ratio with divisible-by-32 dimensions; $T = 32$ evaluation GOP length). Given a standard-compliant 4:2:0 video encoder-decoder pair $\mathcal{C}_f(\cdot; q, \mathbf{\Delta Q}_t)$ from codec family $f \in \{0:\text{HEVC/libx265},\; 1:\text{AVC/libx264}\}$ operating at base Constant Rate Factor $q \in \{27, 32, 37, 42\}$ (mapped to normalized compression severity $s = \operatorname{clip}((q - 22)/25, 0, 1) \in \{0.20, 0.40, 0.60, 0.80\}$), we learn a conditional spatio-temporal pre-editor $\mathcal{P}_\theta$ (`6,108,814` parameters) producing both pre-edited frames $\{I_{\text{edit}, t}\}_{t=0}^{T-1}$ and native $16 \times 16$-pixel Quantization Group (QG) offsets $\{\mathbf{\Delta Q}_t\}_{t=0}^{T-1}$:

$$\min_{\theta} \mathbb{E}_{(f, q, \mathcal{I})} \left[ \mathcal{R}_{\text{I/P}}\!\Big(\mathcal{C}_f\big(\{I_{\text{edit}, t}\}, q, \{\mathbf{\Delta Q}_t\}\big)\Big) + \mathcal{L}_{\text{task}}\!\big(\mathcal{I}, \{I_{\text{edit}, t}\}, \{\hat{I}_t\}\big) + \mathcal{L}_{\text{attn}} + \mathcal{L}_{\text{temp}} + \mathcal{L}_{\text{reg}} \right]$$

where $\hat{I}_t$ is the standard-decoded frame (`yuv420p` $\to$ RGB) consumed without modification by off-the-shelf machine vision models.

### Three Coupled Bottlenecks Addressed by `tscope-v3`
1. **Inter-Frame Residual Flicker & Camera Ego-Motion Breakdown in Spatial/Unaligned Pre-Editors (`F01`):**
   - Still-image pre-editors (`SCOPE` [Dani *et al.*, 2026], `TSW-Pre` [Yang *et al.*, 2024], `PreprocessingICM` [Lu *et al.*, 2024]) process each frame independently, injecting stochastic attention jitter and sensor-grain residual differences across consecutive frames. Conversely, unaligned recurrent features (`v1` bottleneck ConvGRU) collapse under fast camera ego-motion (`MOT17-13` bus camera), while un-cored pixel warping (`v2` `wo_ctu_delta_qp`) leaks camera sensor noise into P-frames. `tscope-v3` couples **6-DOF Cauchy-IRLS global affine motion compensation** (`< 0.75 px` Static Identity Snap), **soft sensor-noise deadzone coring** ($\tau_{\text{core}} = 0.012$), **anti-flicker background attention locking**, and **motion-locked background EMA** across 32-frame GOPs.
2. **Full-Frame Distortion & Virtual-Codec Mismatch (`F04`, `F07`):**
   - Video pre-processors trained purely with full-frame MSE and hand-crafted virtual codecs (Zhao *et al.*, DCC 2024) waste bits on task-irrelevant background texture and suffer proxy-to-encoder mismatch against real `x265`/`x264` RDO and `aq-mode=2`. `tscope-v3` pairs a real `libx265`/`libx264` `yuv420p` forward pass with a multi-codec `8x8` DCT + `16x16` motion-compensated backward Straight-Through Estimator (STE), evaluated via a strictly de-confounded single-factor ablation (`full_tscope` vs. `wo_x265_ste_proxy` at identical $\kappa_{\max} = 4.0$ and $\lambda_{\text{temp}}$).
3. **Boundary Spillover & Zero-FLOP Savings in Codec ROI Coupling (`F02`, `F08`, `F12`, `F17`):**
   - Image pre-editors discard their learned semantic mask $A_t$ prior to encoding, whereas naive block ROI allocation either assigns negative foreground $\Delta\text{QP}$ (starving global bit budgets) or downsamples QG maps into coarse $32 \times 32$-pixel macro-cells (`min(16, H//16)` in `v2`), spilling positive $+\Delta\text{QP}$ onto small pedestrian boundaries (`MOT17-11`). Meanwhile, naive GOP QP reuse (`v2` `iwa_gop_qp_reuse`) still ran the full U-Net on every P-frame ($0\%$ FLOP savings). `tscope-v3` enforces a **`3x3` max-pool foreground halo guard**, **native $16 \times 16$-pixel QG resolution** ($40 \times 24$ on $640 \times 384$), **strictly non-negative background coarsening** ($\Delta Q \in [0, +7.14]$, foreground $\Delta Q = 0$), and a **true inter-frame U-Net bypass (`forward_inter_bypass_step`)** cutting 32-frame GOP compute by **`96.72%` GMACs** (`4,409.07` $\to$ `144.59` GMACs).

---

## 3. Exact Mathematical Formulation (`models/temporal_scope.py`, `video_codecs/hybrid_x265_ste.py`, `losses/vcm_temporal_loss.py`)

### 3.1 Multi-Codec and Polynomial-Hinge Severity Conditioning (`SeverityConditioner`)
Codec family $f \in \{0, 1\}$ (`0`: `libx265` HEVC Main 4:2:0; `1`: `libx264` AVC High/Main 4:2:0) is embedded via $e_f \in \mathbb{R}^8$ (`nn.Embedding(2, 8)`), where both rows receive 50/50 balanced gradient updates during training. Normalized compression severity $s = \operatorname{clip}((q - 22.0)/25.0, 0, 1)$ is expanded into a 6-D polynomial-hinge basis:

$$\phi(s) = \big[s,\; s^2,\; s^3,\; h_{0.50}(s),\; h_{0.75}(s),\; h_{0.90}(s)\big] \in \mathbb{R}^6, \qquad h_\tau(s) = \operatorname{clip}\!\left(\frac{s - \tau}{1 - \tau},\, 0,\, 1\right)$$

A two-layer GELU MLP maps $[e_f, \phi(s)] \in \mathbb{R}^{14} \to \mathbb{R}^{64} \to c \in \mathbb{R}^{64}$. Every encoder/decoder stage `FiLMConvBlock` applies two $3 \times 3$ convolutions with Feature-wise Linear Modulation:

$$H^{(1)} = \operatorname{GELU}\!\big(\mathcal{N}_1(\operatorname{Conv}_{3\times 3}(X))\big), \qquad \operatorname{FiLM}(X, c) = \operatorname{GELU}\!\Big(\mathcal{N}_2\big(\operatorname{Conv}_{3\times 3}(H^{(1)})\big) \odot \big(1 + \gamma(c)\big) + \beta(c)\Big)$$

where full-resolution stages (`enc1`, `dec1` at $H \times W$) set $\mathcal{N}_{1,2} = \operatorname{Identity}$ (`use_norm=False`) to preserve memory bandwidth and linear mode connectivity across sharded checkpoints, while coarser stages (`enc2`, `bottleneck`, `enc3`, `ctx_fuse`, `dec2`) use `GroupNorm(8, C)`.

### 3.2 6-DOF Cauchy-IRLS Global Affine Motion Estimation & `< 0.75 px` Static Identity Snap
For every consecutive frame pair $(I_{t-1}, I_t)$, `estimate_global_affine_irls` executes strictly in FP32 (`torch.amp.autocast(enabled=False)`) on BT.601 luma $G_t = 0.299 I_t^R + 0.587 I_t^G + 0.114 I_t^B$:
1. **Coarse $H/8$ Translation + Parabolic Subpixel Search:** Over integer shifts $(\Delta x, \Delta y) \in [-6, +6]^2$ at resolution $(H/8, W/8)$, we minimize mean $L_1$ error plus zero-motion tie-break $10^{-6}(\Delta x^2 + \Delta y^2)$, followed by 1D parabolic subpixel refinement clamped to $[-0.5, +0.5]$ grid units to initialize normalized affine matrix $\theta_t^{(0)} \in \mathbb{R}^{2 \times 3}$.
2. **Two-Scale Cauchy-Weighted Gauss-Newton IRLS ($H/8$: 4 iters, $H/4$: 3 iters):** At each scale $(H_s, W_s)$ and iteration, $G_{t-1}$ is bilinearly warped via normalized coordinates $(x, y) \in [-1, 1]^2$ to $\tilde{G}_{t-1}$. With residual $e = G_t - \tilde{G}_{t-1}$, interior validity $v = \mathbb{I}(|x| \le 0.96 \wedge |y| \le 0.96)$, symmetric spatial gradients $\bar{I}_x = \frac{1}{2}(\nabla_x G_t + \nabla_x \tilde{G}_{t-1})$, $\bar{I}_y = \frac{1}{2}(\nabla_y G_t + \nabla_y \tilde{G}_{t-1})$, and Cauchy Lorentzian weights ($\delta = 0.04$) that reject moving foreground outliers:
   $$w_{\text{IRLS}} = v \cdot \frac{\delta^2}{\delta^2 + e^2}, \qquad J = \big[\bar{I}_x x,\; \bar{I}_x y,\; \bar{I}_x,\; \bar{I}_y x,\; \bar{I}_y y,\; \bar{I}_y\big]$$
   $$\big(J^\top \operatorname{diag}(w_{\text{IRLS}}) J + 10^{-2} I_6\big)\,\Delta p = J^\top \big(w_{\text{IRLS}} \odot e\big), \qquad \theta_t \leftarrow \theta_t + \operatorname{ reshape}\!\big(\operatorname{clip}(\Delta p, -0.25, +0.25)\big)$$
3. **`< 0.75 px` Corner-Displacement Static Identity Snap:** Let $\{q_k\}_{k=1}^4 = \{(\pm 1, \pm 1, 1)^\top\}$ be the four normalized image corners. Converting corner displacements $\Delta q_k = \theta_t q_k - q_{k, 1:2}$ into pixels $(|\Delta q_{k,x}|\frac{W}{2}, |\Delta q_{k,y}|\frac{H}{2})$, if $\max_{k} \|\Delta q_k\|_{\infty,\text{px}} < 0.75\text{ px}$ (or $\max |G_t - G_{t-1}| < 10^{-5}$), we snap $\theta_t \leftarrow \begin{bmatrix}1 & 0 & 0 \\ 0 & 1 & 0\end{bmatrix}$ (`is_static = True`), preventing subpixel bilinear resampling blur on tripod sequences (`MOT17-02/04/09`).

### 3.3 Sensor-Noise Coring, Multi-Scale `H/8` Context Fusion, Smoothstep STE & Motion-Locked EMA
Given warped previous frame $\tilde{I}_{t-1} = \mathcal{W}_{\theta_t}(I_{t-1})$ and interior validity masks $V_{\text{full}, t} = \mathbb{I}(|\mathcal{G}_{\theta_t}|_\infty \le 0.985)$ (forced to $\mathbf{1}$ when `is_static`), `forward_single_step` computes:
1. **Sensor-Noise Soft Deadzone Coring (`F01` fix):**
   $$\Delta I_{\text{val}, t} = (I_t - \tilde{I}_{t-1}) \odot V_{\text{full}, t}, \qquad \Delta I_{\text{MC}, t} = \operatorname{sign}(\Delta I_{\text{val}, t}) \max\!\big(|\Delta I_{\text{val}, t}| - 0.012,\, 0\big)$$
   suppressing zero-mean camera sensor grain on static P-frame regions before feeding $X_{\text{in}, t} = [I_t, \Delta I_{\text{MC}, t}] \in \mathbb{R}^{6 \times H \times W}$ into `enc1`.
2. **Multi-Scale Encoder + `H/8` Global Context + Motion-Warped `H/4` ConvGRU:**
   $$F_1 = \text{enc1}(X_{\text{in}, t}, c) \in \mathbb{R}^{48 \times H \times W}, \quad F_2 = \text{enc2}(\text{down1}(F_1), c) \in \mathbb{R}^{96 \times \frac{H}{2} \times \frac{W}{2}}, \quad F_3 = \text{bottleneck}(\text{down2}(F_2), c) \in \mathbb{R}^{192 \times \frac{H}{4} \times \frac{W}{4}}$$
   $$F_4 = \text{enc3}(\text{down3}(F_3), c) \in \mathbb{R}^{192 \times \frac{H}{8} \times \frac{W}{8}}, \qquad F_{3,\text{ctx}} = \text{ctx\_fuse}\!\big([F_3, \operatorname{Upsample}_{2\times}(F_4)],\, c\big) \in \mathbb{R}^{192 \times \frac{H}{4} \times \frac{W}{4}}$$
   $$H_t = \operatorname{ConvGRU}\!\big(F_{3,\text{ctx}},\; \mathcal{W}_{\theta_t}(H_{t-1}) \odot V_{H/4, t}\big) \quad (\text{with } H_{t-1} = \mathbf{0} \text{ at } t=0 \text{ or } \texttt{disable\_temporal})$$
   followed by skip-connected decoder stages $U_2 = \text{dec2}([\text{up2}(\operatorname{Up}(H_t)), F_2], c)$ and $U_1 = \text{dec1}([\text{up1}(\operatorname{Up}(U_2)), F_1], c) \in \mathbb{R}^{48 \times H \times W}$.
3. **Smoothstep STE Foreground Protection & Anti-Flicker Background Attention Lock:**
   With logits $Z_{\text{attn}, t} = \text{attn\_head}(U_1) + b_{\text{cond}}(c)$ and $A_t = \sigma(Z_{\text{attn}, t})$:
   $$u_{\text{sat}} = \operatorname{clip}\!\left(\frac{A_t - 0.12}{0.46},\, 0,\, 1\right), \quad A_{\text{hard}, t} = u_{\text{sat}}^2 (3 - 2 u_{\text{sat}}), \quad A_{\text{proxy}, t} = \sigma\!\big(2(Z_{\text{attn}, t} + 0.65)\big)$$
   $$A^{\text{raw}}_t = A_{\text{proxy}, t} + \operatorname{stop\_grad}(A_{\text{hard}, t} - A_{\text{proxy}, t})$$
   Using spatially smoothed motion magnitude $m_t = G_{5\times 5}(\frac{1}{3}\|I_t - \tilde{I}_{t-1}\|_1)$ and static background confidence $C_{\text{bg}, t} = \operatorname{clip}\!\big(G_{5\times 5}\big(V_{\text{full}, t} e^{-28 m_t}\operatorname{clip}(1 - m_t/0.08, 0, 1)\big), 0, 1\big)$, spurious P-frame clutter spikes are suppressed while immediately passing foreground ($A^{\text{raw}}_t \ge 0.45$):
   $$o_{\text{fg}, t} = \operatorname{clip}\!\left(\frac{A^{\text{raw}}_t - 0.45}{0.25},\, 0,\, 1\right), \qquad w_{\text{lock}, t} = 0.70 \, C_{\text{bg}, t} \, (1 - o_{\text{fg}, t}) \operatorname{clip}\!\big(1 - \operatorname{stop\_grad}(\tilde{A}^{\text{eff}}_{t-1}),\, 0,\, 1\big)$$
   $$A^{\text{eff}}_t = (1 - w_{\text{lock}, t}) A^{\text{raw}}_t + w_{\text{lock}, t} \operatorname{stop\_grad}(\tilde{A}^{\text{eff}}_{t-1})$$
4. **Asymmetric Foreground Preservation & Motion-Locked Background EMA:**
   With severity gate $g(s) = 0.30 + 0.70 s$ (`max_obj_res = 0.025`, `max_bg_res = 0.03`):
   $$\Delta_{\text{obj}, t} = \big(0.025 \, g(s)\big) \, A^{\text{eff}}_t \odot \tanh(\text{obj\_res\_head}(U_1)), \qquad I_{\text{obj}, t} = \operatorname{clip}(I_t + \Delta_{\text{obj}, t},\, 0,\, 1)$$
   $$B(I_t, c) = \big(1 - \alpha_{\text{blur}}(c)\big) G_{5\times 5}(I_t) + \alpha_{\text{blur}}(c)\operatorname{Up}_{2\times}\!\big(G_{5\times 5}(\operatorname{AvgPool}_{2\times 2}(G_{5\times 5}(I_t)))\big)$$
   $$\Delta_{\text{bg}, t} = (1 - 0.75 \, C_{\text{bg}, t}) \cdot 0.03 \tanh(\text{bg\_res\_head}(U_1)), \qquad I^{\text{sp}}_{\text{bg}, t} = \operatorname{clip}\!\big(B(I_t, c) + \Delta_{\text{bg}, t},\, 0,\, 1\big)$$
   $$w_{\text{temp}, t} = \operatorname{clip}\!\Big(0.90 \, G_{5\times 5}\!\big(\sigma(\text{bg\_temporal\_gate}(U_1))\big) \, C_{\text{bg}, t},\; 0,\; 0.92\Big), \qquad I_{\text{bg}, t} = w_{\text{temp}, t} \tilde{I}_{\text{bg}, t-1} + (1 - w_{\text{temp}, t}) I^{\text{sp}}_{\text{bg}, t}$$
   $$I_{\text{edit}, t} = A^{\text{eff}}_t \odot I_{\text{obj}, t} + (1 - A^{\text{eff}}_t) \odot I_{\text{bg}, t}$$

### 3.4 Halo-Guarded Strictly Non-Negative QG $\Delta\text{QP}$ Allocation (`compute_ctu_qp_offsets`)
All legacy zero-sum / negative foreground $\Delta Q < 0$ equations (`F02`) and $16 \times 16$ macro-cell downsampling caps (`F08`, `F12`) are eliminated in `tscope-v3`:
1. **`3x3` Foreground Halo Guard & Native $16 \times 16$-Pixel QG Pooling:** At $640 \times 384$ with `qg_size = 16`, the QG grid has exact native dimensions $(H_{\text{QG}}, W_{\text{QG}}) = (24, 40)$ ($960$ blocks of $16 \times 16$ pixels):
   $$A^{\text{guard}}_t = \operatorname{MaxPool}_{3\times 3, \text{stride}=1}(A^{\text{eff}}_t), \qquad \bar{A}^{\text{QG}}_t = \operatorname{AvgPool}_{16\times 16}(A^{\text{guard}}_t), \qquad M^{\text{QG}}_t = \operatorname{MaxPool}_{16\times 16}(A^{\text{guard}}_t)$$
   $$A^{\text{QG}}_t = 0.50 \, \bar{A}^{\text{QG}}_t + 0.50 \, M^{\text{QG}}_t \in [0, 1]^{B \times T \times 24 \times 40}$$
   *(Engineering note: `tscope-v3` tightens the pooling weights from `v2`'s $0.62\,\text{avg} + 0.38\,\text{max}$ to $0.50\,\text{avg} + 0.50\,\text{max}$ on the $3\times 3$-dilated mask so that any single foreground or 1-pixel halo pixel with $A^{\text{guard}}_t \ge 0.76$ yields $A^{\text{QG}}_t \ge 0.38 = \tau$, strictly zeroing $+\Delta\text{QP}$ across object boundaries on `MOT17-11`.)*
2. **Strictly Non-Negative Background Coarsening Above $\tau = 0.38$:**
   $$\mathbf{\Delta Q}^{\text{raw}}_t = \operatorname{clip}\!\left(\kappa_{\max} \cdot 1.75 \cdot \frac{\max(\tau - A^{\text{QG}}_t,\, 0)}{\tau},\; 0.0,\; 0.14 \times 51.0\right) \in [0.00,\, +7.14]$$
   where $\kappa_{\max} = 4.0$ (`max_delta_qp=4.0`, reaching $4.0 \times 1.75 = +7.00$ QP steps on pure background $A^{\text{QG}}_t = 0$). Foreground and boundary-halo QGs ($A^{\text{QG}}_t \ge 0.38$) receive **exact $\mathbf{\Delta Q}_t = 0.0$** (`qoffset = 0`). Passed to `ffmpeg addroi`, active regions with $\mathbf{\Delta Q} \ge 0.15$ emit $q_{\text{offset}} = \mathbf{\Delta Q} / 51.0 \in [0.00, +0.14]$ (horizontal run-length merged within $|q_{\text{offset,next}} - q_{\text{offset}}| \le 0.005$; filtergraphs $> 4{,}000$ chars are offloaded automatically via `-filter_script:v` or `-/vf`). Note that $[-0.18, +0.14]$ in `HybridX265STE` is solely an external safety clamp—`TemporalSCOPE` never emits negative offsets.
3. **Foreground-Safe Monotonic P-Frame Carry-Forward (`full_tscope`):** Within each GOP ($t \bmod K \neq 0$, $K = 32$), background coarsening is carried forward monotonically unless foreground enters the QG cell:
   $$s_{\text{fg}, t} = \max\!\big(A^{\text{QG}}_t,\; 0.90 \, M^{\text{QG}}_t\big), \qquad w_{\text{safe}, t} = 1 - \operatorname{clip}\!\left(\frac{s_{\text{fg}, t} - 0.24}{0.12},\, 0,\, 1\right)$$
   $$\tilde{B}_{t-1} = \max\!\Big(B_{t-1},\; 0.5 B_{t-1} + 0.5 \operatorname{AvgPool}_{3\times 3}(B_{t-1})\Big), \qquad \mathbf{\Delta Q}_t = \operatorname{clip}\!\big(\max(\mathbf{\Delta Q}^{\text{raw}}_t,\, w_{\text{safe}, t} \tilde{B}_{t-1}),\, 0,\, 7.14\big)$$
   with state update $B_t = \max(w_{\text{safe}, t} B_{t-1},\, w_{\text{safe}, t} \mathbf{\Delta Q}_t)$ reset at keyframe boundaries $t \bmod K = 0$.

### 3.5 True FLOP-Saving Long-GOP Inter-Frame Bypass (`forward_inter_bypass_step`, `iwa_gop_qp_reuse`)
To resolve `F17`, when `reuse_iframe_qp=True` (`iwa_gop_qp_reuse`), `TemporalSCOPE` runs the full `6.11M` U-Net + ConvGRU **only on keyframes** ($t \bmod K = 0$, i.e., $1$ frame per $T=32$ GOP). On all $31$ inter-frames ($t \bmod K \neq 0$):
1. `enc1..dec1` and `temporal_gru` are **completely bypassed**.
2. $[I_t, \Delta I_{\text{MC}, t}, \tilde{A}^{\text{eff}}_{t-1}] \in \mathbb{R}^{7 \times H \times W}$ is $2\times$ average-pooled to $(H/2, W/2)$, processed by the zero-initialized 2-layer `inter_refine` head (`Conv3x3(7->16) + GELU + Conv3x3(16->4)`, `1,604` parameters), and bilinearly upsampled to yield $(\delta z_{\text{attn}, t}, \tilde{d}_{\text{obj}, t})$.
3. Newly disoccluded or fast-moving objects are promoted via $p_{\text{mot}, t} = \operatorname{clip}((m_t - 0.045)/0.08, 0, 1)$, updating $Z_{\text{attn}, t} = \operatorname{logit}(\operatorname{clip}(\max(\tilde{A}_{t-1}, 0.80 p_{\text{mot}, t}), 10^{-4}, 1-10^{-4})) + \delta z_{\text{attn}, t}$, followed by Smoothstep STE, bounded $\Delta_{\text{obj}, t}$, $\Delta_{\text{bg}, t} = \mathbf{0}$, and motion-locked background EMA $I_{\text{bg}, t} = w_{\text{temp}, t}\tilde{I}_{\text{bg}, t-1} + (1 - w_{\text{temp}, t})B(I_t, c)$.
4. Keyframe QG offsets $\mathbf{\Delta Q}_0$ are reused statically across the GOP ($\mathbf{\Delta Q}_t \equiv \mathbf{\Delta Q}_0$), collapsing the `ffmpeg` filtergraph to a single static `-vf addroi=...` pass and reducing neural compute on $640 \times 384$, $T = 32$ from **`4,409.07` GMACs (`137.78` GMACs/frame)** to **`144.59` GMACs (`4.52` GMACs/frame)**—an exact **`96.72%` GMAC reduction**.

### 3.6 Multi-Codec Hybrid Forward / Proxy-Backward STE & Single-Teacher Objective
- **Forward Pass (`HybridX265STE`):** Invokes real `ffmpeg` with `-pix_fmt yuv420p` on `libx265` (`family_id=0`, `-x265-params crf=q:bframes=B:keyint=G:min-keyint=G:scenecut=0:aq-mode=2`) or `libx264` (`family_id=1`, `-crf q -x264-params bframes=B:keyint=G:min-keyint=G:scenecut=0:aq-mode=2`), verifying `stream.pix_fmt == "yuv420p"` and matching PTS-ordered `ffprobe` packet sizes to decoded frames.
- **Backward Proxy (`DifferentiableVideoProxy`):** Applies $8 \times 8$ orthonormal 2D DCT quantization on $t=0$ Intra frames and $16 \times 16$ block motion-compensated residuals on $t \ge 1$ Inter frames ($q_{\text{step}} = 0.008(1 + 0.06\,\mathbb{I}_{f=1}) \cdot 2^{(q + \Delta Q_{8\times 8} - 22)/6}$, preserving full $24 \times 40$ QG nearest-neighbor mapping to $8 \times 8$ TUs and $1.18\times$ AVC rate scaling), coupled via STE on both pixels and per-frame I/P rate:
  $$\hat{I}_t = \hat{I}_{\text{proxy}, t} + \operatorname{stop\_grad}(\hat{I}_{\text{real}, t} - \hat{I}_{\text{proxy}, t}), \qquad R_t = R_{\text{proxy}, t} + \operatorname{stop\_grad}(R_{\text{real}, t} - R_{\text{proxy}, t})$$
- **Single-Teacher Loss (`VCMTemporalPreEditLoss`, FP32 outside AMP):** Supervised **exclusively** by frozen `Faster R-CNN ResNet50-FPN v2` (`{p2, p3, p4, p5}`, `use_yolov8s=False`):
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \mathcal{L}_{\text{att}} + \mathcal{L}_{\text{bg}} + \mathcal{L}_{\text{temp}} + \mathcal{L}_{\text{reg}}$$
  1. **Task & Direct Foreground Fidelity ($\mathcal{L}_{\text{task}}$):** Masked RMS-normalized Smooth-$L_1$ ($\beta=0.1$, $\lambda_{\text{det}}=5.0$) across FPN $\{P_2..P_5\}$ on $16$-px padded context mask $M_{\text{det}}$, plus foreground ROI pixel fidelity $\lambda_{\text{fg\_pix}}(L_1 + 10\,\text{MSE})$ ($\lambda_{\text{fg\_pix}}=4.0$), spatial gradient preservation ($\lambda_{\text{fg\_grad}}=2.0$), and post-codec foreground RMSE ($\lambda_{\text{fg\_recon}}=1.5$).
  2. **High-Recall Attention & Saturation ($\mathcal{L}_{\text{att}}$):** Pixel-weighted BCE + Tversky ($\alpha=0.25, \beta=0.75$, $\lambda_{\text{attn}}=0.45$) on $4$-px dilated mask $M_{\text{halo4}} = \operatorname{MaxPool}_{9\times 9}(M_{\text{attn}})$, foreground saturation hinges $\lambda_{\text{fg\_sat}}[\operatorname{ReLU}(0.95 - A_t)_{M_{\text{det}}} + 0.5\operatorname{ReLU}(0.90 - A_t)_{M_{\text{attn}}}]$ ($\lambda_{\text{fg\_sat}}=2.0$), and background sparsity $\lambda_{\text{attn\_sparse}}\|(1 - M_{\text{halo4}})\odot A_t\|_1$ ($\lambda_{\text{attn\_sparse}}=1.4$).
  3. **Spatial Background Simplification & Separated I/P Rate ($\mathcal{L}_{\text{bg}}$):** $\lambda_{\text{bg\_flat}}\|M_{\text{bg}}(I_{\text{edit}} - G_{5\times 5}(I_{\text{edit}}))\|_1 + \lambda_{\text{bg\_smooth}}\|\nabla I_{\text{edit}}\|_{1, M_{\text{bg}}} + \lambda_{\text{rate}, I}\bar{R}_I + \lambda_{\text{rate}, P}\bar{R}_P + \lambda_{\text{rate}, P, \log}\log(1 + \bar{R}_P / 0.05)$ with $(\lambda_{\text{bg\_flat}}, \lambda_{\text{bg\_smooth}}, \lambda_{\text{rate}, I}, \lambda_{\text{rate}, P}, \lambda_{\text{rate}, P, \log}) = (1.2, 0.8, 1.0, 4.5, 0.50)$.
  4. **Motion-Aligned Temporal Pixel, Attention & QG Monotonicity ($\mathcal{L}_{\text{temp}}$):** Occlusion-weighted ($w_{\text{occ}} = e^{-5\|I_t - \tilde{I}_{t-1}\|_1}$) motion-warped background residual $\lambda_{\text{temp\_bg}} \cdot 20 \|M_{\text{bg}, t} w_{\text{occ}} V_{\text{full}}(I_{\text{edit}, t} - \tilde{I}_{\text{edit}, t-1})\|_1$ ($\lambda_{\text{temp\_bg}}=1.0$ in active `VARIANT_CONFIGS`, `0.0` in `spatial_only_scope`), object residual smoothness $\lambda_{\text{temp\_obj}} \cdot 5\|\Delta_{\text{obj}, t} - \Delta_{\text{obj}, t-1}\|_1$ ($\lambda_{\text{temp\_obj}}=0.5$), background attention anti-flicker $\lambda_{\text{temp\_attn}}\|M_{\text{bg}, t} M_{\text{bg}, t-1}(A^{\text{eff}}_t - A^{\text{eff}}_{t-1})\|_1$ ($\lambda_{\text{temp\_attn}}=0.8$), and QG background monotonicity/smoothness ($\lambda_{\text{dqp\_mono}}=0.4$, $\lambda_{\text{dqp\_smooth}}=0.15$).
  5. **Residual & QG Foreground Halo Guard Regularization ($\mathcal{L}_{\text{reg}}$):** $\lambda_{\text{res\_l1}}(\|\Delta_{\text{obj}}\|_1 + \|\Delta_{\text{bg}}\|_1) + \lambda_{\text{res\_tv}}(\operatorname{TV}(\Delta_{\text{obj}}) + \operatorname{TV}(\Delta_{\text{bg}})) + \lambda_{\text{dqp\_fg}}\|M^{\text{QG}}_{\text{fg\_guard}} \odot \operatorname{ReLU}(\mathbf{\Delta Q})\|_1$ with $(\lambda_{\text{res\_l1}}, \lambda_{\text{res\_tv}}, \lambda_{\text{dqp\_fg}}) = (0.1, 0.1, 1.0)$.

---

## 4. Module-Level Parameter & Analytical GMAC Breakdown

| Module (`models/temporal_scope.py`) | Spatial Scale | Norm | Parameters | Role |
| :--- | :---: | :---: | ---: | :--- |
| `conditioner` (`SeverityConditioner`) | Global $1\times 1$ | — | `5,136` | `Embedding(2, 8)` + 6-D $\phi(s)$ $\to$ 2-layer GELU MLP (`64`-D) |
| `enc1` + `down1` | $H \to H/2$ | Identity | `103,488` | Full-res `[I_t, diff_mc_t]` FiLM conv (`29,664`) + stride-2 down (`73,824`) |
| `enc2` + `down2` | $H/2 \to H/4$ | GroupNorm(8) | `474,048` | Half-res FiLM conv (`178,944`) + stride-2 down (`295,104`) |
| `bottleneck` + `down3` + `enc3` + `ctx_fuse` | $H/4 \leftrightarrow H/8$ | GroupNorm(8) | `2,990,784` | Bottleneck (`689,664`) + $H/8$ global context branch (`1,279,680`) + fusion (`1,021,440`) |
| `temporal_gru` (`TemporalConvGRUCell`) | $H/4 \times W/4$ | — | `1,991,232` | Motion-warped 192-ch recurrent alignment (`3x3` reset/update/candidate) |
| `up2` + `dec2` + `up1` + `dec1` | $H/4 \to H$ | GN(8) / Id | `537,936` | Multi-scale FiLM skip decoder (`dec1` unnormalized at $H\times W$) |
| Prediction heads (`attn`, `obj`, `bg`, `blur`, `gate`) | $H \times W$ | — | `4,586` | $A_t$, $\Delta_{\text{obj}}$, $\Delta_{\text{bg}}$, adaptive blur mix, background EMA gate |
| `inter_refine` (`forward_inter_bypass_step`) | $H/2 \times W/2$ | — | `1,604` | Zero-init $7\to 16\to 4$ P-frame warp refinement (`iwa_gop_qp_reuse`) |
| **Total (`TemporalSCOPE` v3)** | **$640 \times 384$, $T=32$** | — | **`6,108,814`** | **`137.78` GMACs/f (`full_tscope`) \| `4.52` GMACs/f (`iwa_gop_qp_reuse`, `-96.72%`)** |

---

## 5. Q1 Experimental Protocol, Pre-Registered Falsification Criteria & Zero-Drift Sync Pipeline

### 5.1 Datasets, Codec CTC Settings & De-Confounded 5-Variant Matrix
- **Primary Split (`MOT17`, Disjoint Sequences):** Train on `MOT17-02, 04, 05, 10`; validate Uniform & Greedy Model Soups + constituent-shard linear mode connectivity on `MOT17-09` (`524` SHA-256-verified frames under a uniform real-`libx265` single-teacher criterion, `F15`/`F20`); evaluate held-out test on `MOT17-11` (`28` clips = `896` frames, tail `897..900` excluded) and `MOT17-13` (`23` clips = `736` frames, tail `737..750` excluded) = **`51` contiguous 32-frame GOP clips (`1,632` frames)** at aspect-preserving $640 \times 384$ ($16:9$).
- **Secondary Zero-Shot VCM Datasets (`F11`, `F25`):** `MOT20` crowded pedestrian tracking (`ismailelbouknify/mot-20`) and multi-class `KITTI` autonomous driving (`leducnhuan/kitti-tracking`, COCO classes `person`, `cyclist`, `car`, `truck` with `DontCare` ignore handling), plus native 8-bit planar `yuv420p` `.yuv` support for `SFU-HW-Objects-v1` / HEVC Class B (`minhngt02/hevc-b`).
- **Codec Operating Grid (`F05`, `F13`, `F14`):** `CRFS = [27, 32, 37, 42]`, `-pix_fmt yuv420p`, `-preset medium`, `aq-mode=2`, `gop_len=32` under Low-Delay P (`bframes=0`, primary) and Random Access (`bframes=4`, `IBBP`), across `libx265` (`family_id=0`) and `libx264` (`family_id=1`).
- **Strictly Balanced, De-Confounded 5-Variant Ablation (`3 shards x 5 variants = 15` Kaggle T4 training kernels, `seed=2026`, `28` cosine epochs with `cosine_t_max=18`, `warmup=2`, `patience=6`, `min_epochs=10`):**
  1. `full_tscope`: 6-DOF IRLS MC + ConvGRU + real codec STE + $\kappa_{\max}=4.0$ + `fg_safe` P-carry ($\lambda_{\text{temp}} = 1.0 / 0.5$).
  2. `wo_ctu_delta_qp`: Identical to `full_tscope`, toggling **only** $\kappa_{\max} = 0.0$ (isolates halo-guarded QG $\Delta\text{QP}$ + P-carry).
  3. `spatial_only_scope`: `disable_temporal=True`, $\kappa_{\max} = 0.0$, $\lambda_{\text{temp}} = 0.0$, **with `use_real_x265_ste=True`** (de-confounded framewise baseline vs. `wo_ctu_delta_qp`, fixing `F07`).
  4. `iwa_gop_qp_reuse`: `reuse_iframe_qp=True`, $\kappa_{\max} = 4.0$, `use_real_x265_ste=True` (activates `forward_inter_bypass_step` + static keyframe QG reuse across $T=32$).
  5. `wo_x265_ste_proxy`: Identical to `full_tscope` ($\kappa_{\max} = 4.0$, $\lambda_{\text{temp}} = 1.0 / 0.5$), toggling **only** `use_real_x265_ste=False` (cleanly isolates real-codec STE vs. proxy-only training).

### 5.2 Statistical Inference & Pre-Registered Claim Falsification (`C01..C05`, `F14`, `F15`, `F18`)
- **Monotonic PCHIP BDBR & 10,000-Resample Paired Clip-Block Bootstrap:** Rate-accuracy curves are integrated analytically in $\log_{10}(\text{bpp})$ via shape-preserving monotonic Piecewise Cubic Hermite Interpolating Polynomials (`scipy.interpolate.PchipInterpolator`, per JCT-VC `JCTVC-E137`), paired with exact vectorized $10{,}000$-resample multinomial clip-block bootstrap $95\%$ confidence intervals (`_PreparedClipBootstrapCurve`).
- **Pre-Registered Falsification Thresholds (`CLAIM_FALSIFICATION_CRITERIA` in `experiment_protocol.py`):** Structural readiness (`eligibility.final_ready`) is decoupled from hypothesis testing; each claim requires point estimate $< 0\%$ **and** upper $95\%$ bootstrap CI $< 0\%$:
  - **`C01`:** `full_tscope` vs. `anchor_x265` on Total `FRCNN AP@50` BDBR (`CI95_upper < 0%`).
  - **`C02`:** `wo_ctu_delta_qp` vs. `spatial_only_scope` on 32-frame **Inter/P-frame** `FRCNN AP@50` BDBR (`bdbr_p_gt_ap50_pct`, directly testing whether motion-compensated temporal pre-editing lowers steady-state P-frame rate).
  - **`C03`:** `full_tscope` vs. `wo_ctu_delta_qp` on Total `FRCNN AP@50` BDBR (testing native $16\times 16$ halo-guarded QG `quantOffsets`).
  - **`C04`:** `full_tscope` vs. `wo_x265_ste_proxy` on Total `FRCNN AP@50` BDBR (testing Hybrid Real-Codec STE under identical hyperparameters).
  - **`C05`:** Single-teacher `full_tscope` vs. `anchor_x265` on **unseen zero-shot `YOLOv8s` AP@50** Total BDBR (`bdbr_tot_yolo_gt_ap50_pct`).

### 5.3 Manuscript Structure & Zero-Drift Automated Table Synchronization (`scripts/sync_paper_tables.py`)
- **Manuscript (`paper/main.tex`, 12–13 pages `IEEEtran`):**
  - **Sec. I:** Introduction & Figure 1 (`docs/figures/fig_pipeline_overview.pdf`).
  - **Sec. II:** Related Work & Problem Analysis (`Lu'24`, `Yang'24`, `Zhao'24`, `Bajić'25`, `Fernández-Menduiña'26`, `Dani'26`).
  - **Sec. III:** Proposed `T-SCOPE v3` Framework + Algorithm 1 + Table I (`6,108,814` Parameter & GMAC Specification).
  - **Sec. IV:** MPEG-VCM CTC Experimental Protocol, De-Confounded Ablations & Pre-Registered Falsification Design.
  - **Sec. V:** Experimental Results (`\input{metrics_macros.tex}` + `\input{tables_generated.tex}`: Tables II–VI covering SOTA Baselines, 5-Variant Ablations + Per-Sequence + Claim Audit, Zero-Shot Cross-Backbone / `libx264` / RA, `ByteTrack` Tracking + `MOT20`/`KITTI`, and T4 GPU / CPU Codec Complexity).
  - **Sec. VI:** Limitations, Engineering Progression (`v1 -> v2 -> v3`), and Threats to Validity.
  - **Sec. VII:** Conclusion.
- **Single-Command Deterministic Sync (`F09` Elimination):** Running `python scripts/sync_paper_tables.py` ingests `outputs_kaggle_pool/merged_benchmarks.json`, `cross_backbone_bdbr.json`, and `sota_and_tracking_benchmarks.json` (or `sota_and_tracking_report.json`) to regenerate `paper/metrics_macros.tex`, `paper/tables_generated.tex`, `docs/V3_Q1_BENCHMARK_REPORT.md`, and the `README.md` benchmark section atomically with zero human copy-paste.
