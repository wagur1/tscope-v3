import shutil
import time
import torch
from models.temporal_scope import TemporalSCOPE
from video_codecs.hybrid_x265_ste import HybridX265STE, detect_ffmpeg_filter_script_flag
from losses.vcm_temporal_loss import VCMTemporalPreEditLoss


def verify_pipeline() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running T-SCOPE v3 verification on device: {device}")

    model = TemporalSCOPE(
        base_ch=48,
        num_families=2,
        cond_dim=64,
        qg_size=16,
        max_delta_qp=4.0,
        keyframe_stride=32,
    ).to(device)
    codec_ste = HybridX265STE(use_real_x265_in_forward=False).to(device)
    criterion = VCMTemporalPreEditLoss().to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"T-SCOPE v3 trainable parameters: {num_params:,} ({num_params / 1e6:.3f} M)")

    # Simulate batch of 16:9 widescreen clips: B=2, T=4 frames, 384x640 (24x40 QGs at qg_size=16)
    b, t, h, w = 2, 4, 384, 640
    video = torch.rand((b, t, 3, h, w), device=device)
    family_id = torch.tensor([0, 1], device=device, dtype=torch.long)  # 0=libx265, 1=libx264
    severity = torch.tensor([0.20, 0.60], device=device, dtype=torch.float32)  # CRF 27, CRF 37

    m_attn = torch.zeros((b, t, 1, h, w), device=device)
    m_attn[:, :, :, 96:256, 160:320] = 1.0
    m_det = torch.zeros((b, t, 1, h, w), device=device)
    m_det[:, :, :, 80:272, 144:336] = 1.0

    t0 = time.perf_counter()
    out = model(video, family_id, severity, reuse_iframe_qp=False)
    codec_out = codec_ste(out["i_edit"], severity, out["ctu_qp_offsets"], family_id=family_id)

    i_clean_flat = video.view(b * t, 3, h, w)
    i_recon_flat = codec_out["video_recon"].view(b * t, 3, h, w)
    teacher_fpn = {
        "p2": torch.nn.functional.avg_pool2d(i_clean_flat, 4).repeat(1, 16, 1, 1).detach(),
        "p3": torch.nn.functional.avg_pool2d(i_clean_flat, 8).repeat(1, 16, 1, 1).detach(),
    }
    student_fpn = {
        "p2": torch.nn.functional.avg_pool2d(i_recon_flat, 4).repeat(1, 16, 1, 1),
        "p3": torch.nn.functional.avg_pool2d(i_recon_flat, 8).repeat(1, 16, 1, 1),
    }

    losses = criterion(
        i_edit=out["i_edit"],
        i_recon=codec_out["video_recon"],
        attn_map=out["attn_map"],
        eff_attn_map=out["eff_attn_map"],
        delta_obj=out["delta_obj"],
        delta_bg=out["delta_bg"],
        rate_proxy_bpp=codec_out["rate_proxy_bpp"],
        i_frame_bpp_proxy=codec_out["i_frame_bpp_proxy"],
        p_frame_bpp_proxy=codec_out["p_frame_bpp_proxy"],
        ctu_qp_offsets=out["ctu_qp_offsets"],
        i_orig=video,
        aligned_prev_edit=out["aligned_prev_edit"],
        aligned_prev_orig=out["aligned_prev_orig"],
        warp_valid_mask=out["warp_valid_mask"],
        m_attn=m_attn,
        m_det=m_det,
        teacher_fpn_feats=teacher_fpn,
        student_fpn_feats=student_fpn,
    )

    losses["loss_total"].backward()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Verify both codec family embeddings received gradients (F10)
    fam_grad = model.conditioner.family_emb.weight.grad
    assert fam_grad is not None and bool((fam_grad.abs().sum(dim=1) > 0).all()), (
        "Both family_id=0 (libx265) and family_id=1 (libx264) embeddings must receive gradients!"
    )

    # Verify iwa_gop_qp_reuse P-frame bypass skips U-Net on inter-frames (F17)
    model.eval()
    with torch.no_grad():
        out_iwa = model(video[:1], family_id[:1], severity[:1], reuse_iframe_qp=True)
        stats_iwa = dict(model.last_forward_stats)
        mac_32_full = model.estimate_gmacs(384, 640, clip_len=32, reuse_iframe_qp=False)
        mac_32_iwa = model.estimate_gmacs(384, 640, clip_len=32, reuse_iframe_qp=True)
    assert stats_iwa["full_unet_steps"] == 1 and stats_iwa["bypass_inter_steps"] == t - 1
    assert out_iwa["ctu_qp_offsets"].shape == (1, t, 24, 40)

    # Real ffmpeg yuv420p multi-codec smoke check if available (F05, F10, F12, F13)
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        real_codec = HybridX265STE(use_real_x265_in_forward=True, preset="ultrafast")
        smoke_clip = out["i_edit"][:1, :4].detach()
        smoke_dqp = out["ctu_qp_offsets"][:1, :4].detach()
        enc_x265 = real_codec.evaluate_real_x265(
            smoke_clip, qp=32, ctu_qp_offsets=smoke_dqp, family_id=0, gop_len=32, bframes=4
        )
        enc_x264 = real_codec.evaluate_real_x265(
            smoke_clip, qp=32, ctu_qp_offsets=smoke_dqp, family_id=1, gop_len=32, bframes=0
        )
        assert enc_x265["pix_fmt"] == "yuv420p" and enc_x264["pix_fmt"] == "yuv420p"
        print(
            f"Real codec smoke OK (filter_script_flag={detect_ffmpeg_filter_script_flag()}): "
            f"x265(RA b=4)={enc_x265['total_bpp']:.4f} bpp ({enc_x265['frame_types']}) | "
            f"x264(LDP b=0)={enc_x264['total_bpp']:.4f} bpp ({enc_x264['frame_types']})"
        )

    grad_norm = sum(
        p.grad.norm().item() for p in model.parameters() if p.grad is not None
    )
    print(f"Output shapes:")
    print(f"  i_edit         : {tuple(out['i_edit'].shape)}")
    print(f"  attn_map       : {tuple(out['attn_map'].shape)}")
    print(
        f"  ctu_qp_offsets : {tuple(out['ctu_qp_offsets'].shape)} "
        f"(min={out['ctu_qp_offsets'].min():.2f}, max={out['ctu_qp_offsets'].max():.2f})"
    )
    print(f"  video_recon    : {tuple(codec_out['video_recon'].shape)}")
    print(
        f"  32f GOP GMACs  : full_tscope={mac_32_full['total_gmacs']:.2f} GMACs "
        f"({mac_32_full['gmacs_per_frame']:.2f}/f) vs iwa_gop_qp_reuse={mac_32_iwa['total_gmacs']:.2f} GMACs "
        f"({mac_32_iwa['gmacs_per_frame']:.2f}/f)"
    )
    print(
        f"Losses: total={losses['loss_total'].item():.4f} | "
        f"task={losses['loss_task'].item():.4f} | "
        f"att={losses['loss_att'].item():.4f} | "
        f"bg={losses['loss_bg'].item():.4f} | "
        f"temp={losses['loss_temp'].item():.4f} | "
        f"reg={losses['loss_reg'].item():.4f}"
    )
    print(f"Backward pass verified! Total grad norm: {grad_norm:.4f} | Step time: {elapsed_ms:.1f} ms")


if __name__ == "__main__":
    verify_pipeline()
