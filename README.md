# T-SCOPE v3: Tiền Biên Tập Video Nhất Quán Thời Gian và Phân Bổ Lượng Tử Hóa Cấp CTU cho Video Coding for Machines (`x265` / `x264`)

Mã nguồn PyTorch, giao thức thực nghiệm chuẩn Q1 (IEEE TCSVT / TIP), bộ kiểm thử tự động (`55/55` unit tests) và bản thảo `IEEEtran` 12 trang cho **T-SCOPE v3**.

- **Bản thảo bài báo LaTeX:** [`paper/main.tex`](./paper/main.tex) và [`paper/main.pdf`](./paper/main.pdf) (đồng bộ tự động từ JSON thực nghiệm qua [`scripts/sync_paper_tables.py`](./scripts/sync_paper_tables.py)).
- **Trọng số phát hành (5 Uniform Model Soups đã kiểm định trên `MOT17-09`):** [`release_ckpts/`](./release_ckpts/) và báo cáo tính liên thông tuyến tính [`docs/soup_connectivity_report.json`](./docs/soup_connectivity_report.json).
- **Báo cáo đánh giá đầy đủ chuẩn Q1:** [`docs/V3_Q1_BENCHMARK_REPORT.md`](./docs/V3_Q1_BENCHMARK_REPORT.md).

---

## 1. Tổng hợp những thay đổi cốt lõi từ T-SCOPE v2 lên T-SCOPE v3

`T-SCOPE v3` được thiết kế lại toàn diện từ `v2` nhằm khắc phục triệt để **26 điểm hạn chế phương pháp luận và thực nghiệm (F01–F26)** theo tiêu chuẩn thẩm định khắt khe của các tạp chí Q1 (**IEEE TCSVT / IEEE TIP**):

| Hạng mục kỹ thuật | Hạn chế ở `T-SCOPE v2` | Nâng cấp chuẩn hóa ở `T-SCOPE v3` |
| :--- | :--- | :--- |
| **1. Định dạng màu (Chroma Subsampling)** | Đầu vào PNG RGB khiến `ffmpeg` tự chuyển ngầm sang `yuv444p`, làm phình bitrate nền của anchor và lệch hoàn toàn chuẩn nén thực tế. | Ép cứng `-pix_fmt yuv420p` chuẩn MPEG VCM / HEVC CTC trên toàn bộ phép nén `libx265` và `libx264` trong [`video_codecs/hybrid_x265_ste.py`](./video_codecs/hybrid_x265_ste.py). |
| **2. Cấu trúc GOP & Chế độ mã hóa** | Dùng `keyint=30` không tường minh `bframes` (mặc định `bframes=4`) nhưng mô tả nhầm là Low-Delay P. | Chuẩn hóa `EVAL_GOP_LEN = 32`, tách rõ 2 chế độ độc lập: **Low-Delay P (`bframes = 0`)** và **Random Access (`IBBP`, `bframes = 4`)**. |
| **3. Đa họ Codec (`SeverityConditioner`)** | Chỉ cố định 1 họ codec (`libx265`), không kiểm chứng khả năng khái quát sang bộ mã hóa khác. | Mở rộng nhúng đa họ codec (`num_families = 2`: `family_id=0` cho **HEVC/`libx265`**, `family_id=1` cho **H.264/`libx264`**) và đánh giá chéo trực tiếp. |
| **4. Tỉ lệ khung hình & Dải điểm làm việc CRF** | Resize vuông `512×512` làm méo ngang người đi bộ `1.78×` (từ chuẩn `16:9`); dải `CRF ∈ {22, 27, 32, 37}` thiếu vùng bitrate thấp khiến đường cong mAP ít chồng lấp khi tính BD-Rate. | Giữ nguyên tỉ lệ `16:9` chuẩn **`640×384` (`IMG_H=384, IMG_W=640`)** và dịch dải đo sang **`CRF ∈ {27, 32, 37, 42}`** phủ trọn vùng suy giảm mAP của chuẩn VCM. |
| **5. Độ phân giải lưới `addroi` CTU/QG & Viền bảo vệ** | Bóp bản đồ tầm quan trọng xuống tối đa `16×16` ô trước khi gán `addroi`, làm mất chi tiết sub-CTU và gây nhiễu halo từ khối nền QP cao lan vào biên vật thể. | Gỡ bỏ hoàn toàn nút thắt downsample `2×`, giữ nguyên lưới Quantization Group `16×16 px` (**`40×24 = 960` ô QG** tại `640×384`, mịn gấp `3.75×` so với `v2`) kèm lớp đệm **`3×3` Foreground Halo Guard** trước khi gom QG. |
| **6. Bypass P-frame thực sự (`iwa_gop_qp_reuse`)** | Chế độ GOP reuse ở `v2` vẫn chạy toàn bộ mạng U-Net/ConvNeXt trên mọi frame rồi mới trộn đầu ra (`0%` giảm FLOPs, thiếu bảng đo FPS/VRAM). | Cài đặt đường tắt thực sự `forward_inter_bypass_step()` trong [`models/temporal_scope.py`](./models/temporal_scope.py): chỉ chạy mạng đầy đủ tại I-frame, còn `31` P-frame chỉ warp nhẹ theo chuyển động affine 6-DOF, **giảm `96.78%` GMACs/frame** (`137.71 -> 4.43 GMACs`) và **tăng tốc `20.4×` trên Tesla T4 (`5.55 -> 113.21 FPS`)**. |
| **7. Ổn định số học FP16/AMP** | Trộn xác suất biên `1 - 1e-4` dưới `float16` bị làm tròn thành `1.0`, gây `torch.logit(1.0) = +inf` và `NaN` trong `binary_cross_entropy` ở các epoch sâu. | Ép tính toán nhánh trộn logit ở `FP32` (`autocast(enabled=False)`), chặn an toàn `nan_to_num` trước `BCE` trong [`losses/vcm_temporal_loss.py`](./losses/vcm_temporal_loss.py) và tự động fallback batch sang `FP32` nếu gặp gradient không hữu hạn. |
| **8. Chống rò rỉ Detector (True Zero-Shot)** | Đưa cả `Faster R-CNN` lẫn `YOLOv8s` vào hàm loss lúc huấn luyện nhưng lại tuyên bố đánh giá "zero-shot cross-architecture" trên `YOLOv8s`. | **Huấn luyện với duy nhất 1 teacher đóng băng (`Faster R-CNN ResNet50-FPN v2`, `use_yolov8s = False`)**. Toàn bộ **4 kiến trúc còn lại** (`YOLOv8s`, `RetinaNet`, `FCOS`, `SSDLite-MobileNetV3`) hoàn toàn ẩn lúc train, đảm bảo kiểm chứng zero-shot 100% trung thực. |
| **9. Công bằng hóa Ablation & Cân bằng Shard** | Tắt `real_x265_ste` ở các baseline nhưng bật ở `full_tscope` (gây nhiễu biến), đồng thời chia số lượng shard không đều giữa các biến thể (`5` vs `2` shard). | Cân bằng tuyệt đối **`3 shards × 5 variants = 15 shards`**. Bật công bằng `use_real_x265_ste = True` cho cả 4 biến thể chính (`full_tscope`, `wo_ctu_delta_qp`, `spatial_only_scope`, `iwa_gop_qp_reuse`) và tách riêng biến thể thứ 5 `wo_x265_ste_proxy` để cô lập chính xác đóng góp của codec-in-the-loop STE. |
| **10. Thống kê BD-Rate chuẩn Q1 & Kiểm định giả thuyết** | Dùng đa thức bậc 3 dễ dao động Runge khi ít điểm chồng lấp; thiếu khoảng tin cậy thống kê và thiếu tiêu chí bác bỏ giả thuyết. | Dùng nội suy đơn điệu từng khúc **`scipy.interpolate.PchipInterpolator`** trên trục `log10(bpp)`, tách riêng **Total BDBR** và **Inter/P-Frame BDBR**, kèm **`10,000` lần paired clip-bootstrap 95% CI** và **5 tiêu chí kiểm định tiền đăng ký `C01–C05`** trong [`scripts/experiment_protocol.py`](./scripts/experiment_protocol.py). |
| **11. Đối chứng SOTA ngoại vi, Đa tập dữ liệu & Tracking** | Chỉ tự so với biến thể nội bộ trên tập đơn lẻ `MOT17` ở tác vụ phát hiện vật thể. | Tích hợp trực tiếp mã nguồn & trọng số công khai của **SCOPE (`miccunifi/SCOPE`)**; bổ sung bài toán **Multi-Object Tracking (ByteTrack: `MOTA`, `IDF1`, `HOTA`)**; mở rộng đánh giá sang **MOT20**, **KITTI Tracking** và chuẩn YUV **HEVC CTC Class B**. |
| **12. Đồng bộ tự động Code – Bài báo & Tái lập** | Công thức trong bản thảo `v2` bị lệch tham số so với code thực chạy; manifest chứa đường dẫn tuyệt đối của máy cá nhân. | Viết mới bản thảo `IEEEtran` 12 trang khớp 100% code `v3`; [`scripts/sync_paper_tables.py`](./scripts/sync_paper_tables.py) tự sinh macro/bảng LaTeX từ JSON; chuẩn hóa đường dẫn tương đối an toàn trong manifest. |

---

## 2. Đối chiếu các hướng tiếp cận VCM trước đây

| # | Công trình & Nơi công bố | Cơ chế chính | Giới hạn kỹ thuật mà T-SCOPE v3 giải quyết |
| :--- | :--- | :--- | :--- |
| **1** | **Video Preprocessing for Machine Vision**<br>*(Fei Zhao et al., IEEE DCC 2024)* | Tiền xử lý 2 nhánh (2D + 3D Conv) trước `x264`/`x265`, huấn luyện qua Virtual Codec với MSE toàn khung. | MSE toàn khung giữ lại nhiễu nền dư thừa; chỉ học qua proxy ảo nên lệch phân phối lượng tử của `x265` thật (`v3` chứng minh trực tiếp qua đối chứng `full_tscope` vs `wo_x265_ste_proxy`). |
| **2** | **SCOPE**<br>*(Silvia Dani et al., 2025/2026 — `miccunifi/SCOPE`)* | Điều biến FiLM theo `(f, s)`, trộn nhánh bảo toàn vật thể và nhánh làm mờ nền qua bản đồ chú ý `A`. | Xử lý độc lập từng ảnh gây nhấp nháy nền liên khung làm tăng phần dư P-frame; không truyền bản đồ chú ý xuống bộ phân bổ QP cấp CTU/QG của codec (`v3` đối chứng trực tiếp với trọng số gốc SCOPE và `spatial_only_scope`). |
| **3** | **TSW-Pre**<br>*(Mingyi Yang et al., IEEE TCSVT 2024)* | U-Net kết hợp Task-Attentive Semantic Modulation (SFT) và chưng cất đặc trưng FPN. | Đòi hỏi chạy mạng detector nặng ngay tại phía bộ mã hóa ở mọi khung hình; không hỗ trợ điều biến theo mức QP hoặc bỏ qua khung liên tiếp. |
| **4** | **Preprocessing Enhanced ICM**<br>*(Guo Lu et al., IEEE TCSVT 2024)* | Hybrid Forward-BPG / Backward-Proxy với lớp thích ứng lượng tử QAL. | Chỉ thiết kế cho nén ảnh tĩnh Intra, không giải quyết bù chuyển động toàn cục hay kế thừa bản đồ QP trên chuỗi GOP video. |
| **5** | **WA-RDO / IWA-RDO**<br>*(S. Fernández-Menduiña et al., IEEE ICASSP 2026)* | Xấp xỉ Jacobian miền đặc trưng; tái sử dụng bản đồ trọng số I-frame cho toàn bộ GOP. | Tái sử dụng tĩnh không bù chuyển động bị lệch vị trí khi camera hoặc vật thể dịch chuyển nhanh (`v3` khắc phục bằng warp 6-DOF Cauchy-IRLS tốc độ `741.84 FPS` ở chế độ `iwa_gop_qp_reuse`). |

---

## 3. Kiến trúc và Cơ chế T-SCOPE v3

**T-SCOPE v3** (`6,108,814` tham số) đặt phía trước bộ mã hóa chuẩn (`libx265` / `libx264`) hoàn toàn tương thích luồng bit chuẩn (mọi bộ giải mã chuẩn đều giải mã trực tiếp không cần sửa đổi):

```text
Chuỗi Video {I_t} (640×384, 16:9)
       │
       ▼
[TemporalSCOPE v3 (6,108,814 params)]
  │  • SeverityConditioner đa họ codec (HEVC x265 / AVC x264) & mức độ nén s
  │  • 6-DOF Cauchy-IRLS Affine Motion Alignment (FP32, 741.84 FPS) + Static Snap
  │  • Chế độ đầy đủ (full_tscope): Multi-Scale FiLM U-Net + H/4 ConvGRU
  │  • Chế độ siêu tốc (iwa_gop_qp_reuse): Bypass P-frame (-96.78% GMACs, 113.21 FPS)
  ├──► 1. Khung hình tiền biên tập {I_edit,t} (Smoothstep STE FG + Occlusion-Gated EMA BG)
  └──► 2. Bản đồ chú ý hiệu dụng + 3×3 Halo Guard ──► Lưới ΔQP 40×24 QG (16×16 px)
                                                               │
                                                               ▼
5 Detectors + ByteTrack ◄── Giải mã chuẩn ◄── libx265 / libx264 (-pix_fmt yuv420p, addroi)
```

### 3.1. Điều kiện hóa đa họ codec và mức độ nén (`SeverityConditioner`)
- Nhúng rời rạc họ bộ mã hóa `f ∈ {0: libx265, 1: libx264}` (`num_families = 2`, 8 chiều) ghép với khai triển phi tuyến 6 chiều của mức nén chuẩn hóa `s = clip((QP - 22) / 25, 0, 1)`, tạo vector điều biến `c ∈ R^64` cho toàn bộ các khối `FiLMConvBlock`.

### 3.2. Căn chỉnh chuyển động 6-DOF Cauchy-IRLS, Khóa nền EMA & Bypass P-frame
- **6-DOF Cauchy-IRLS (`estimate_global_affine_irls`):** Tìm kiếm thô tại `H/8` và tinh chỉnh lặp trọng số Cauchy tại `H/8 -> H/4` bằng `float32` kèm *Static Identity Snap* (`< 0.75 px`), căn chỉnh chính xác `I_{t-1}`, trạng thái ẩn `h_{t-1}` và nền tích lũy `bg_{t-1}` trên các cảnh lia máy nhanh (`MOT17-13`, `KITTI`).
- **Chế độ `iwa_gop_qp_reuse` (`forward_inter_bypass_step`):** Chỉ chạy bộ mã hóa đặc trưng trên khung đầu GOP (`t = 0`), còn các P-frame kế tiếp warp trực tiếp bản đồ chú ý và nền từ khung trước qua ma trận affine 6-DOF (tính toán logit ở `FP32` chống tràn số), cắt bỏ `96.78%` khối lượng tính toán mỗi frame.

### 3.3. Phân bổ lượng tử hóa QG `40×24` có đệm bảo vệ `3×3` (`compute_ctu_qp_offsets`)
- Trước khi gom xuống lưới Quantization Group `qg_size = 16` (`40×24` vùng tại `640×384`), bản đồ tiền cảnh đi qua bộ lọc cực đại `3×3` (*Foreground Halo Guard*) nhằm ngăn khối nền tăng QP sát rìa làm hỏng gradient biên vật thể.
- Tín hiệu `qg_attn = 0.62 * avg_qg + 0.38 * max_qg` điều khiển độ lệch lượng tử nền không âm `ΔQP ∈ [0, 4.0]` (ánh xạ sang `addroi qoffset` có gộp run-length ngang) và kế thừa an toàn qua các P-frame trong GOP.

### 3.4. Huấn luyện Hybrid Real-Codec STE một Teacher duy nhất (`Single-Teacher`)
- Nhánh forward gọi trực tiếp `ffmpeg libx265 -pix_fmt yuv420p` để mô hình thấy đúng nhiễu lượng tử và dự đoán liên khung thật; nhánh backward truyền đạo hàm qua `DifferentiableVideoProxy`.
- Hàm mất mát [`VCMTemporalPreEditLoss`](./losses/vcm_temporal_loss.py) chỉ dùng **duy nhất `Faster R-CNN ResNet50-FPN v2`** đóng băng (`{p2, p3, p4, p5}`), kết hợp mất mát bảo toàn ROI người đi bộ, Tversky + BCE cho mặt nạ chú ý, ước lượng bitrate tách riêng I/P và ràng buộc ổn định nền liên khung.

---

## 4. Cấu trúc mã nguồn và Quy trình thực thi

```text
├── README.md
├── TCSVT_Paper_Blueprint.md            # Bản thiết kế toán học & giao thức thực nghiệm v3
├── paper/                              # Bản thảo bài báo IEEEtran 12 trang
│   ├── main.tex                        # Nguồn LaTeX chính (nạp tự động macros & tables)
│   ├── main.pdf                        # Bản PDF biên dịch hoàn chỉnh
│   ├── references.bib                  # Tài liệu tham khảo BibTeX
│   ├── metrics_macros.tex              # Macro số liệu sinh tự động từ JSON
│   └── tables_generated.tex            # Các bảng kết quả LaTeX sinh tự động
├── docs/
│   ├── V3_Q1_BENCHMARK_REPORT.md       # Báo cáo chi tiết toàn bộ bảng thực nghiệm Q1
│   ├── soup_connectivity_report.json   # Kiểm định Linear Mode Connectivity trên MOT17-09
│   └── figures/                        # Sơ đồ kiến trúc & đường cong Rate-Accuracy
├── models/
│   ├── temporal_scope.py               # TemporalSCOPE v3 (6.11M params), IRLS, P-frame bypass, Halo Guard
│   └── frozen_detector_fpn.py          # Trình bao Faster R-CNN FPN & đánh giá đa detector theo chunk
├── video_codecs/
│   └── hybrid_x265_ste.py              # Hybrid libx265/libx264 (yuv420p, LDP/RA GOP=32) + Differentiable Proxy
├── losses/
│   └── vcm_temporal_loss.py            # Hàm mất mát VCM Single-Teacher ổn định FP32/AMP
├── vcm_datasets/
│   └── vcm_video_datasets.py           # Bộ nạp dữ liệu MOT17, MOT20, KITTI Tracking, HEVC Class B (640×384)
├── release_ckpts/                      # 5 Uniform Model Soups đã xác thực SHA-256 & Mode Connectivity
├── scripts/
│   ├── experiment_protocol.py          # Hợp đồng giao thức schema_version=3 & kiểm định giả thuyết C01–C05
│   ├── kaggle_pool_orchestrator.py     # Điều phối song song đa tài khoản Kaggle T4
│   ├── kaggle_shard_runner.py          # Huấn luyện & đánh giá phân mảnh (train / eval_soup)
│   ├── validate_soups.py               # Kiểm định Uniform Soup vs. Greedy Soup vs. Constituent trên MOT17-09
│   ├── eval_cross_backbone_bdbr.py     # Đánh giá 5 kiến trúc detector, libx264 & Random Access (IBBP)
│   ├── eval_sota_and_tracking.py       # Đối chứng SCOPE gốc, ByteTrack (MOTA/IDF1/HOTA), MOT20/KITTI & FPS/GMACs
│   ├── merge_shards.py                 # Gộp Model Soup & tính PCHIP BDBR + 10,000 Bootstrap 95% CI
│   └── sync_paper_tables.py            # Đồng bộ tự động JSON -> LaTeX, docs và README
└── tests/                              # 55 bài kiểm thử đơn vị bảo vệ hợp đồng toán học & dữ liệu
```

### Quy trình tái lập từ đầu đến cuối

```powershell
# 1. Chạy toàn bộ 55 bài kiểm thử hợp đồng kỹ thuật v3
pytest -q

# 2. Huấn luyện cân bằng 15 shard (3 shards × 5 variants, 640×384, Single-Teacher Faster R-CNN)
python -I scripts/kaggle_pool_orchestrator.py --pool_file "<path/to/pool.json>" --action push --run_mode train --ablation_sweep --expected_shards 15 --git_revision "<40_CHAR_SHA>"

# 3. Thu thập shard, tạo 5 Uniform Model Soups và kiểm định Mode Connectivity trên tập val MOT17-09
python -I scripts/kaggle_pool_orchestrator.py --pool_file "<path/to/pool.json>" --action pull --run_mode train
python -I scripts/merge_shards.py --action soup --manifest "<train_manifest.json>" --ckpt_dir ./outputs_kaggle_pool/prepared_soups_v3 --expected_shards 15
python -I scripts/kaggle_pool_orchestrator.py --pool_file "<path/to/pool.json>" --action push --run_mode validate_soups --git_revision "<40_CHAR_SHA>"

# 4. Chạy song song 3 cụm đánh giá Phase 3 (eval_soup + eval_cross_backbone + eval_sota_and_tracking)
python -I scripts/kaggle_pool_orchestrator.py --pool_file "<path/to/pool.json>" --action push --run_mode eval_soup --expected_shards 12 --git_revision "<PHASE3_SHA>"
python -I scripts/kaggle_pool_orchestrator.py --pool_file "<path/to/pool.json>" --action push --run_mode eval_cross_backbone --expected_shards 1 --git_revision "<PHASE3_SHA>"
python -I scripts/kaggle_pool_orchestrator.py --pool_file "<path/to/pool.json>" --action push --run_mode eval_sota_and_tracking --expected_shards 4 --sota_task_split "sota,tracking,second_dataset,complexity" --extra_dataset_slugs ismailelbouknify/mot-20 leducnhuan/kitti-tracking --git_revision "<PHASE3_SHA>"

# 5. Tổng hợp PCHIP BDBR + 10,000 Bootstrap 95% CI và tự động đồng bộ LaTeX + Báo cáo + README
python -I scripts/merge_shards.py --action eval --manifest "<eval_manifest.json>" --eval_dir ./outputs_kaggle_pool/eval_v3_final --ckpt_dir ./release_ckpts --expected_shards 12
python -I scripts/sync_paper_tables.py --pool-dir ./outputs_kaggle_pool
```

---

<!-- BEGIN_V3_BENCHMARKS -->
## 5. Verified T-SCOPE v3 Q1 Benchmark Summary (Auto-Synced via `scripts/sync_paper_tables.py`)

Source manifest SHA-256: `pending` (`schema_version = 3`, `640x384` 16:9, `yuv420p`, `GOP = 32`, `CRFS = [27, 32, 37, 42]`, `PchipInterpolator` + `10,000` paired clip-bootstrap 95% CIs). Full disaggregation in [`docs/V3_Q1_BENCHMARK_REPORT.md`](./docs/V3_Q1_BENCHMARK_REPORT.md).

*(Đang tự động cập nhật ngay khi cụm kernel Phase 3 hoàn tất gộp `merged_benchmarks.json`)*
<!-- END_V3_BENCHMARKS -->

---

## 6. Kiểm định Linear Mode Connectivity & Hiệu năng phần cứng Tesla T4 (Đã hoàn tất)

### 6.1. Kiểm định Model Soup trên tập Validation `MOT17-09` ([`docs/soup_connectivity_report.json`](./docs/soup_connectivity_report.json))

Đo trực tiếp với luồng nén thật `libx265 -pix_fmt yuv420p` tại `640×384` (`held_out_test_used = false`, không chọn lọc trên tập test):

| Biến thể (`release_ckpts/tscope_soup_<variant>.pt`) | Uniform Soup Loss | Trung bình ± Độ lệch chuẩn 3 Shard lẻ | Greedy Soup Loss (Shard chọn) | Liên thông tuyến tính (`Uniform <= Mean`) |
| :--- | :---: | :---: | :---: | :---: |
| **`full_tscope`** | **`3.7816`** | `3.9308 ± 0.0876` | `3.7816` (`[10, 0, 5]`) | **Đạt (`-0.1492`)** |
| **`iwa_gop_qp_reuse`** | **`3.8312`** | `3.9621 ± 0.0683` | `3.8051` (`[8, 13]`) | **Đạt (`-0.1309`)** |
| **`wo_ctu_delta_qp`** | **`3.7605`** | `3.8211 ± 0.0145` | `3.7272` (`[11, 1]`) | **Đạt (`-0.0606`)** |
| **`wo_x265_ste_proxy`** | **`3.7010`** | `3.8383 ± 0.1087` | `3.7010` (`[4, 9, 14]`) | **Đạt (`-0.1374`)** |
| `spatial_only_scope` | `3.9920` | `3.9760 ± 0.0967` | `3.8933` (`[12]`) | `Không` |

### 6.2. Độ phức tạp tính toán & Tốc độ thực tế trên GPU NVIDIA Tesla T4 (`640×384`, `GOP = 32`)

| Cấu hình / Thành phần | Tham số | GMACs / Frame | Độ trễ TB (`ms/frame`) | Độ trễ P95 (`ms/frame`) | Tốc độ (`FPS`) | VRAM Đỉnh (`MB`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `full_tscope` (Chạy đầy đủ mọi frame) | `6.11M` | `137.71` | `180.05` | `--` | `5.55 FPS` | `1336.78` |
| `spatial_only_scope` (Độc lập từng frame) | `6.11M` | `137.71` | `174.49` | `--` | `5.73 FPS` | `~1336` |
| **`iwa_gop_qp_reuse` (Bypass P-frame `v3`)** | **`6.11M`** | **`4.43` (`-96.78%`)** | **`8.83`** | **`9.08`** | **`113.21 FPS` (`20.4×`)** | **`1325.53`** |
| Riêng khối `6-DOF Cauchy-IRLS Motion` | `0` | `--` | `1.35` | `--` | **`741.84 FPS`** | `342.63` |
