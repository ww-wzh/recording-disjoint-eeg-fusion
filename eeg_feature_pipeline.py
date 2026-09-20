"""Sanitized eight-channel EEG loading and feature extraction.

The implementation is mechanically extracted from the frozen experiment source.
Data paths are relative or supplied through CBSF_DATA_ROOT; no raw data are bundled.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos
import torch

DEFAULT_SFREQ = 250
WIN_SEC = 8.5
STRIDE_SEC = 0.5
NUM_CLASSES = 4
CACHE_FEATURE_VERSION = 19
BANDS = [
    (0.5, 4.0),
    (4.0, 8.0),
    (8.0, 12.0),
    (12.0, 18.0),
    (18.0, 30.0),
    (30.0, 55.0),
]

# Frozen feature configuration used to produce the Route A predictions.
# Keep these values explicit so a clean checkout cannot silently fall back to
# a different feature representation.
NOTCH_LINE_FREQ_HZ = 50.0
NOTCH_LINE_WIDTH_HZ = 1.5
NOTCH_LINE_ATTEN = 1.0
NOTCH_Q = 30.0
APPLY_RECORDING_NOTCH = True
NORMALIZE_WITHIN_WINDOW = False
USE_BANDPASS_FILTER = True
BANDPASS_LOW_HZ = 0.5
BANDPASS_HIGH_HZ = 55.0
BANDPASS_ORDER = 4
WORKLOAD_THETA_ALPHA_REL_BOOST = 1.0
USE_DIFFERENTIAL_ENTROPY = True
USE_INTER_CHANNEL_CORR = False
FRONTAL_FEATURE_MODE = "first_half"
PREFRONTAL_CHANNEL_INDICES = None
FAA_LEFT_RIGHT_ALPHA = None
FRONTAL_REL_POWER_WEIGHT = 1.0
EXPECTED_EEG_CHANNELS = 8
EXPECTED_FEATURE_DIM = 272

DATA_ROOT = Path(
    os.environ.get("CBSF_DATA_ROOT", Path(__file__).resolve().parent / "data" / "raw_data")
).resolve()
_FILE_STEMS = {0: "natural", 1: "lowlevel", 2: "midlevel", 3: "highlevel"}
SUBJECTS = [
    {
        "id": subject,
        "arithmetic": {
            label: str(DATA_ROOT / "Arithmetic_Data" / f"{stem}-{subject}.txt")
            for label, stem in _FILE_STEMS.items()
        },
        "stroop": {
            label: str(DATA_ROOT / "Stroop_Data" / f"{stem}-{subject}.txt")
            for label, stem in _FILE_STEMS.items()
        },
    }
    for subject in range(1, 16)
]


def normalize_subjects(subjects_raw: List[Dict]) -> List[Dict]:
    if not isinstance(subjects_raw, list) or len(subjects_raw) == 0:
        raise ValueError('SUBJECTS must be a non-empty list')
    first = subjects_raw[0]
    if isinstance(first, dict) and 'id' in first and ('arithmetic' in first) and ('stroop' in first):
        return subjects_raw
    if len(subjects_raw) == 1 and isinstance(subjects_raw[0], dict):
        top = subjects_raw[0]
        subj_keys = [k for k in top.keys() if isinstance(k, str) and re.match('^subject_\\d+$', k)]
        if subj_keys:
            out: List[Dict] = []
            label_keys_by_label = {0: ['baseline', 'natural'], 1: ['low_stress', 'low', 'lowlevel'], 2: ['mid_stress', 'mid', 'midlevel'], 3: ['high_stress', 'high', 'highlevel']}
            for sk in sorted(subj_keys, key=lambda x: int(x.split('_')[1])):
                sid = int(sk.split('_')[1])
                blob = top[sk]
                if not isinstance(blob, dict) or 'arithmetic' not in blob or 'stroop' not in blob:
                    raise ValueError(f'Invalid subject blob under key {sk}')

                def task_to_numeric(task_blob: Dict) -> Dict[int, str]:
                    task_numeric: Dict[int, str] = {}
                    for lab, key_candidates in label_keys_by_label.items():
                        found = None
                        for cand in key_candidates:
                            if cand in task_blob:
                                found = task_blob[cand]
                                break
                        if found is None:
                            raise KeyError(f'Missing label for subject {sid}: label={lab} candidates={key_candidates}')
                        task_numeric[lab] = found
                    return task_numeric
                out.append({'id': sid, 'arithmetic': task_to_numeric(blob['arithmetic']), 'stroop': task_to_numeric(blob['stroop'])})
            return out
    raise ValueError("Unsupported SUBJECTS format. Expect either internal format (list of per-subject dicts with 'id') or your format ([{ 'subject_1': {...}, ... }]).")

def _is_int_like(arr: np.ndarray, tol: float=1e-06) -> bool:
    if arr.size == 0:
        return False
    return np.all(np.abs(arr - np.round(arr)) < tol)

def _infer_time_column_and_sfreq(col: np.ndarray) -> Tuple[bool, Optional[float]]:
    """
    If first column is true time in seconds:
    - monotonic increasing
    - step small enough for typical sfreq
    """
    if col.size < 20:
        return (False, None)
    d = np.diff(col)
    if not np.all(d > 0):
        return (False, None)
    d_med = float(np.median(d))
    d_mean = float(np.mean(d))
    d_std = float(np.std(d))
    if d_med <= 0 or d_med > 0.05:
        return (False, None)
    if d_mean == 0:
        return (False, None)
    if d_std / d_mean > 0.2:
        return (False, None)
    sfreq = 1.0 / d_med
    if sfreq < 50 or sfreq > 2000:
        return (False, None)
    return (True, sfreq)

def _infer_sample_index_column(col: np.ndarray) -> bool:
    """
    If first column is sample index:
    - integer-like
    - strictly increasing
    - step ~ 1
    """
    if col.size < 20:
        return False
    if not _is_int_like(col, tol=0.001):
        return False
    d = np.diff(col)
    if not np.all(d > 0):
        return False
    d_med = float(np.median(d))
    d_mean = float(np.mean(d))
    d_std = float(np.std(d))
    if d_mean == 0:
        return False
    if d_med < 0.5 or d_med > 2.0:
        return False
    if d_std / max(d_mean, 1e-12) > 0.2:
        return False
    return True

def load_txt_eeg(path: str, default_sfreq: float=DEFAULT_SFREQ) -> Tuple[np.ndarray, float]:
    if not os.path.exists(path):
        raise FileNotFoundError(f'TXT not found: {path}')
    rows: List[List[float]] = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or ',' not in line:
                continue
            flts: List[float] = []
            for tok in line.split(','):
                tok = tok.strip()
                if not tok:
                    continue
                try:
                    flts.append(float(tok))
                except ValueError:
                    continue
            if len(flts) >= 2:
                rows.append(flts)
    if not rows:
        raise ValueError(f'No numeric rows parsed from TXT: {path}')
    n_cols = max((len(r) for r in rows))
    arr = np.full((len(rows), n_cols), np.nan, dtype=np.float32)
    for i, r in enumerate(rows):
        arr[i, :len(r)] = np.array(r, dtype=np.float32)
    arr = arr[:, ~np.isnan(arr).all(axis=0)]
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f'Parsed EEG has invalid shape {arr.shape} from {path}')
    col_median = np.nanmedian(arr, axis=0)
    keep_cols = np.ones(arr.shape[1], dtype=bool)
    keep_cols &= np.abs(col_median) < 10000000.0
    if keep_cols.sum() >= 2:
        arr = arr[:, keep_cols]
    first_col = arr[:, 0].astype(np.float64)
    is_time, sfreq_est = _infer_time_column_and_sfreq(first_col)
    if is_time:
        return (arr[:, 1:], float(sfreq_est) if sfreq_est is not None else float(default_sfreq))
    if _infer_sample_index_column(first_col):
        return (arr[:, 1:], float(default_sfreq))
    return (arr, float(default_sfreq))

def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()

def extract_features_from_eeg_windows(eeg: np.ndarray, sfreq: float, win_sec: float, stride_sec: float, bands: List[Tuple[float, float]], device: torch.device, cache_key: str, cache_dir: str, chunk_size: int=64) -> Tuple[torch.Tensor, torch.Tensor]:
    os.makedirs(cache_dir, exist_ok=True)
    feat_cache_path = os.path.join(cache_dir, f'{cache_key}.npz')
    if os.path.exists(feat_cache_path):
        cached = np.load(feat_cache_path)
        fe = torch.from_numpy(cached['features'].astype(np.float32))
        if 'covs' in cached.files:
            co = torch.from_numpy(cached['covs'].astype(np.float32))
        else:
            raise RuntimeError(f"Stale cache missing 'covs': delete {feat_cache_path} or bump CACHE_FEATURE_VERSION")
        return (fe, co)
    if eeg.ndim != 2:
        raise ValueError('EEG must be 2D array [T, C]')
    T, C = eeg.shape
    if C != EXPECTED_EEG_CHANNELS:
        raise ValueError(
            f'Frozen pipeline expects {EXPECTED_EEG_CHANNELS} EEG channels, got {C}'
        )
    if bool(globals().get('APPLY_RECORDING_NOTCH', True)):
        notch_frequency = float(globals().get('NOTCH_LINE_FREQ_HZ', 50.0))
        quality = float(globals().get('NOTCH_Q', 30.0))
        normalized = notch_frequency / (float(sfreq) / 2.0)
        if not 0.0 < normalized < 1.0:
            raise ValueError(f'Sampling rate {sfreq} does not support the frozen 50-Hz notch')
        numerator, denominator = iirnotch(w0=normalized, Q=quality)
        eeg = sosfiltfilt(tf2sos(numerator, denominator), eeg, axis=0).astype(np.float32)
    if bool(globals().get('USE_BANDPASS_FILTER', False)):
        _bp_lo = float(globals().get('BANDPASS_LOW_HZ', 0.5))
        _bp_hi = float(globals().get('BANDPASS_HIGH_HZ', 55.0))
        _bp_ord = int(globals().get('BANDPASS_ORDER', 4))
        nyq = sfreq / 2.0
        if _bp_lo > 0 and _bp_hi < nyq:
            sos = butter(_bp_ord, [_bp_lo / nyq, _bp_hi / nyq], btype='band', output='sos')
            eeg = sosfiltfilt(sos, eeg, axis=0).astype(np.float32)
    win_samples = int(round(win_sec * sfreq))
    stride_samples = int(round(stride_sec * sfreq))
    if win_samples <= 0 or stride_samples <= 0:
        raise ValueError('win_sec/stride_sec too small')
    if T < win_samples:
        raise ValueError(f'EEG length {T} shorter than win_samples {win_samples}')
    n_windows = 1 + (T - win_samples) // stride_samples
    if n_windows <= 0:
        raise ValueError('No windows created')
    n_fft = _next_pow2(win_samples)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sfreq)
    band_masks = []
    for f0, f1 in bands:
        band_masks.append((freqs >= f0) & (freqs < f1))
    hann = torch.from_numpy(np.hanning(win_samples).astype(np.float32)).to(device)
    eps = 1e-12
    features_list: List[np.ndarray] = []
    cov_list: List[np.ndarray] = []
    starts = [i * stride_samples for i in range(n_windows)]
    for chunk_start in range(0, n_windows, chunk_size):
        chunk_end = min(n_windows, chunk_start + chunk_size)
        chunk_starts = starts[chunk_start:chunk_end]
        chunk = np.stack([eeg[s:s + win_samples].T for s in chunk_starts], axis=0).astype(np.float32)
        x_raw = torch.from_numpy(chunk).to(device)
        if bool(globals().get('NORMALIZE_WITHIN_WINDOW', False)):
            ch_mean = x_raw.mean(dim=-1, keepdim=True)
            ch_std = x_raw.std(dim=-1, unbiased=False, keepdim=True).clamp_min(1e-06)
            x = (x_raw - ch_mean) / ch_std
        else:
            x = x_raw
        xw = x * hann[None, None, :]
        X = torch.fft.rfft(xw, n=n_fft, dim=-1)
        power = (X.real ** 2 + X.imag ** 2) / float(n_fft)
        _nf = float(globals().get('NOTCH_LINE_FREQ_HZ', 0.0))
        _nw = float(globals().get('NOTCH_LINE_WIDTH_HZ', 1.5))
        _nat = float(globals().get('NOTCH_LINE_ATTEN', 1.0))
        if _nf > 1.0 and _nw > 0 and (_nat < 1.0):
            nm = (freqs >= _nf - _nw) & (freqs <= _nf + _nw)
            if nm.any():
                nm_t = torch.from_numpy(nm.astype(np.bool_)).to(device)
                power[:, :, nm_t] = power[:, :, nm_t] * _nat
        total_mask = (freqs >= 0.5) & (freqs < 55.0)
        total_mask_t = torch.from_numpy(total_mask.astype(np.bool_)).to(device)
        total_power = power[:, :, total_mask_t].mean(dim=-1, keepdim=True).clamp_min(eps)
        abs_bp_list = []
        rel_bp_list = []
        for m in band_masks:
            m_t = torch.from_numpy(m.astype(np.bool_)).to(device)
            if m_t.any():
                bp = power[:, :, m_t].mean(dim=-1)
            else:
                bp = torch.zeros(power.shape[0], power.shape[1], device=device, dtype=power.dtype)
            abs_bp_list.append(torch.log(bp + eps))
            rel_bp_list.append(bp / total_power.squeeze(-1))
        abs_bp = torch.stack(abs_bp_list, dim=-1)
        rel_bp = torch.stack(rel_bp_list, dim=-1)
        _w_pf = float(globals().get('FRONTAL_REL_POWER_WEIGHT', 1.0))
        _fm_w = str(globals().get('FRONTAL_FEATURE_MODE', 'none'))
        if abs(_w_pf - 1.0) > 1e-06 and rel_bp.shape[1] >= 2:
            _ix_w: Optional[torch.Tensor] = None
            if _fm_w == 'first_half':
                _ix_w = torch.arange(0, rel_bp.shape[1] // 2, device=device, dtype=torch.long)
            elif _fm_w == 'indices':
                _pf = globals().get('PREFRONTAL_CHANNEL_INDICES')
                if _pf is not None:
                    _ix_w = torch.as_tensor(_pf, device=device, dtype=torch.long)
                    _ix_w = _ix_w[(_ix_w >= 0) & (_ix_w < rel_bp.shape[1])]
            if _ix_w is not None and _ix_w.numel() > 0:
                lw = float(np.log(_w_pf))
                abs_bp = abs_bp.clone()
                rel_bp = rel_bp.clone()
                abs_bp[:, _ix_w, :] = abs_bp[:, _ix_w, :] + lw
                rel_bp[:, _ix_w, :] = rel_bp[:, _ix_w, :] * _w_pf
        _tab = float(globals().get('WORKLOAD_THETA_ALPHA_REL_BOOST', 1.0))
        if _tab > 1.0 and rel_bp.shape[-1] > 2:
            rel_bp = rel_bp.clone()
            rel_bp[:, :, 1] *= _tab
            rel_bp[:, :, 2] *= _tab
        ps = power[:, :, total_mask_t]
        ps = ps / ps.sum(dim=-1, keepdim=True).clamp_min(eps)
        spec_ent = -(ps * torch.log(ps + eps)).sum(dim=-1)
        theta = rel_bp[:, :, 1]
        alpha = rel_bp[:, :, 2]
        beta = (rel_bp[:, :, 3] + rel_bp[:, :, 4]).clamp_min(eps)
        ratio_tb = (theta / beta).unsqueeze(-1)
        ratio_ab = (alpha / beta).unsqueeze(-1)
        mean = x.mean(dim=-1)
        std = x.std(dim=-1, unbiased=False)
        rms = torch.sqrt((x ** 2).mean(dim=-1) + eps)
        ptp = x.amax(dim=-1) - x.amin(dim=-1)
        abs_bp_flat = abs_bp.reshape(abs_bp.shape[0], -1)
        rel_bp_flat = rel_bp.reshape(rel_bp.shape[0], -1)
        spec_ent_flat = spec_ent.reshape(spec_ent.shape[0], -1)
        ratio_flat = torch.cat([ratio_tb, ratio_ab], dim=-1).reshape(spec_ent.shape[0], -1)
        time_flat = torch.cat([mean, std, rms, ptp], dim=-1)
        line_len = torch.abs(x[..., 1:] - x[..., :-1]).mean(dim=-1)
        line_flat = line_len.reshape(line_len.shape[0], -1)
        alp_ch = rel_bp[:, :, 2]
        n_ch = alp_ch.shape[1]
        if n_ch >= 2:
            hch = n_ch // 2
            asym_alp = (alp_ch[:, :hch].mean(dim=1) - alp_ch[:, hch:].mean(dim=1)).unsqueeze(-1)
        else:
            asym_alp = torch.zeros(alp_ch.shape[0], 1, device=device, dtype=alp_ch.dtype)
        delta_th = (rel_bp[:, :, 0] + rel_bp[:, :, 1]).clamp_min(eps)
        alp_bet = (rel_bp[:, :, 2] + rel_bp[:, :, 3] + rel_bp[:, :, 4]).clamp_min(eps)
        smr_ratio = (delta_th / alp_bet).unsqueeze(-1)
        smr_flat = smr_ratio.reshape(smr_ratio.shape[0], -1)
        phys_extra = torch.cat([line_flat, asym_alp, smr_flat], dim=-1)
        d1 = x[..., 1:] - x[..., :-1]
        d2 = d1[..., 1:] - d1[..., :-1]
        var0 = x.var(dim=-1, unbiased=False).clamp_min(eps)
        var1 = d1.var(dim=-1, unbiased=False).clamp_min(eps)
        var2 = d2.var(dim=-1, unbiased=False).clamp_min(eps)
        mobility = torch.sqrt(var1 / var0)
        complexity = torch.sqrt(var2 / var1) / mobility.clamp_min(eps)
        hj_act = torch.log(var0 + eps)
        hjorth_flat = torch.cat([hj_act, mobility, complexity], dim=-1).reshape(spec_ent.shape[0], -1)
        gam_ch = rel_bp[:, :, -1].clamp_min(eps)
        bet_ch = (rel_bp[:, :, 3] + rel_bp[:, :, 4]).clamp_min(eps)
        alp_ch2 = rel_bp[:, :, 2].clamp_min(eps)
        th_ch2 = rel_bp[:, :, 1].clamp_min(eps)
        log_ratio_ch = torch.stack([torch.log((alp_ch2 / bet_ch).mean(dim=1)), torch.log((gam_ch / alp_ch2).mean(dim=1)), torch.log((th_ch2 / alp_ch2).mean(dim=1))], dim=-1)
        rel_mu = rel_bp.mean(dim=1)
        rel_sigma = rel_bp.std(dim=1, unbiased=False)
        abs_mu = abs_bp.mean(dim=1)
        se_m = spec_ent.mean(dim=1, keepdim=True)
        se_s = spec_ent.std(dim=1, unbiased=False, keepdim=True)
        spatial_flat = torch.cat([rel_mu, rel_sigma, abs_mu, se_m, se_s], dim=-1)
        mw_parts: List[torch.Tensor] = []
        n_ch_mw = int(rel_bp.shape[1])
        _fm = str(globals().get('FRONTAL_FEATURE_MODE', 'none'))
        _ix_pf: Optional[torch.Tensor] = None
        if _fm == 'first_half' and n_ch_mw >= 2:
            _ix_pf = torch.arange(0, n_ch_mw // 2, device=device, dtype=torch.long)
        elif _fm == 'indices':
            _pf = globals().get('PREFRONTAL_CHANNEL_INDICES')
            if _pf is not None:
                _ix_pf = torch.as_tensor(_pf, device=device, dtype=torch.long)
                _ix_pf = _ix_pf[(_ix_pf >= 0) & (_ix_pf < n_ch_mw)]
        if _ix_pf is not None and _ix_pf.numel() > 0:
            th_m = rel_bp[:, _ix_pf, 1].mean(dim=1, keepdim=True)
            bt_m = (rel_bp[:, _ix_pf, 3] + rel_bp[:, _ix_pf, 4]).mean(dim=1, keepdim=True).clamp_min(eps)
            al_m = rel_bp[:, _ix_pf, 2].mean(dim=1, keepdim=True).clamp_min(eps)
            mw_parts.append(torch.log((th_m / bt_m).clamp_min(eps)))
            mw_parts.append(torch.log(al_m))
        _faa = globals().get('FAA_LEFT_RIGHT_ALPHA')
        if _faa is not None and len(_faa) == 2:
            _L, _R = (int(_faa[0]), int(_faa[1]))
            if 0 <= _L < n_ch_mw and 0 <= _R < n_ch_mw:
                mw_parts.append(torch.log((rel_bp[:, _L, 2] / rel_bp[:, _R, 2].clamp_min(eps)).clamp_min(eps)).unsqueeze(-1))
        de_parts: List[torch.Tensor] = []
        if bool(globals().get('USE_DIFFERENTIAL_ENTROPY', False)):
            for bi, (f0, f1) in enumerate(bands):
                m_b = band_masks[bi]
                m_t = torch.from_numpy(m_b.astype(np.bool_)).to(device)
                if m_t.any():
                    band_sig = torch.fft.irfft(X * m_t[None, None, :].float(), n=n_fft, dim=-1)[..., :win_samples]
                    band_var = band_sig.var(dim=-1, unbiased=False).clamp_min(eps)
                    de = 0.5 * torch.log(2.0 * math.pi * math.e * band_var)
                    de_parts.append(de)
        corr_flat: Optional[torch.Tensor] = None
        if bool(globals().get('USE_INTER_CHANNEL_CORR', False)) and x.shape[1] >= 2:
            _nc_corr = int(x.shape[1])
            x_centered = x - x.mean(dim=-1, keepdim=True)
            x_norm = x_centered / x_centered.norm(dim=-1, keepdim=True).clamp_min(eps)
            corr_mat = torch.bmm(x_norm, x_norm.transpose(1, 2))
            triu_idx = torch.triu_indices(_nc_corr, _nc_corr, offset=1)
            corr_flat = corr_mat[:, triu_idx[0], triu_idx[1]]
        _feat_blocks: List[torch.Tensor] = [abs_bp_flat, rel_bp_flat, spec_ent_flat, ratio_flat, time_flat, spatial_flat, hjorth_flat, phys_extra, log_ratio_ch]
        if mw_parts:
            _feat_blocks.append(torch.cat(mw_parts, dim=-1))
        if de_parts:
            de_all = torch.stack(de_parts, dim=-1)
            _feat_blocks.append(de_all.reshape(de_all.shape[0], -1))
            _feat_blocks.append(de_all.mean(dim=1))
        if corr_flat is not None:
            _feat_blocks.append(corr_flat)
        feats = torch.cat(_feat_blocks, dim=-1)
        n_ch = int(x.shape[1])
        c_cov = torch.matmul(x, x.transpose(1, 2)) / float(x.shape[-1])
        eye = torch.eye(n_ch, device=device, dtype=x.dtype).unsqueeze(0).expand(c_cov.shape[0], -1, -1)
        c_cov = c_cov + float(0.0001) * eye
        features_list.append(feats.detach().cpu().numpy().astype(np.float32))
        cov_list.append(c_cov.detach().cpu().numpy().astype(np.float32))
    features_np = np.concatenate(features_list, axis=0)
    covs_np = np.concatenate(cov_list, axis=0)
    if features_np.shape[1] != EXPECTED_FEATURE_DIM:
        raise RuntimeError(
            f'Frozen feature schema drifted: expected {EXPECTED_FEATURE_DIM}, '
            f'got {features_np.shape[1]}'
        )
    np.savez_compressed(feat_cache_path, features=features_np, covs=covs_np)
    return (torch.from_numpy(features_np), torch.from_numpy(covs_np))

@dataclass(frozen=True)
class DataRecord:
    subject_id: int
    task: str
    label: int
    file_path: str

def build_records(subjects: List[Dict], task: str) -> List[DataRecord]:
    if task not in ('arithmetic', 'stroop'):
        raise ValueError('task must be arithmetic or stroop')
    records: List[DataRecord] = []
    for subj in subjects:
        sid = int(subj['id'])
        task_dict = subj[task]
        for label in range(NUM_CLASSES):
            if label not in task_dict:
                raise KeyError(f'Missing label {label} for subject {sid} task {task}')
            records.append(DataRecord(subject_id=sid, task=task, label=label, file_path=task_dict[label]))
    return records

def _make_cache_key(file_path: str, sfreq: float, win_sec: float, stride_sec: float, bands: List[Tuple[float, float]]) -> str:
    key_obj = {'path': os.path.abspath(file_path), 'sfreq': float(sfreq), 'win_sec': float(win_sec), 'stride_sec': float(stride_sec), 'bands': bands, 'version': CACHE_FEATURE_VERSION}
    return hashlib.md5(json.dumps(key_obj, sort_keys=True).encode('utf-8')).hexdigest()

class EEGWindowDataset(torch.utils.data.Dataset):

    def __init__(self, records: List[DataRecord], win_sec: float, stride_sec: float, bands: List[Tuple[float, float]], default_sfreq: float, cache_dir: str, device_for_feature: torch.device):
        super().__init__()
        self.records = records
        self.win_sec = float(win_sec)
        self.stride_sec = float(stride_sec)
        self.bands = bands
        self.default_sfreq = float(default_sfreq)
        self.cache_dir = cache_dir
        self.device_for_feature = device_for_feature
        feats_all: List[torch.Tensor] = []
        covs_all: List[torch.Tensor] = []
        labels_all: List[torch.Tensor] = []
        subjects_all: List[torch.Tensor] = []
        segments_all: List[torch.Tensor] = []
        self.feature_dim: Optional[int] = None
        self.cov_n_channels: Optional[int] = None
        segment_id = 0
        for rec in records:
            eeg, sfreq = load_txt_eeg(rec.file_path, default_sfreq=self.default_sfreq)
            cache_key = _make_cache_key(rec.file_path, sfreq, self.win_sec, self.stride_sec, self.bands)
            feats, covs = extract_features_from_eeg_windows(eeg=eeg, sfreq=sfreq, win_sec=self.win_sec, stride_sec=self.stride_sec, bands=self.bands, device=self.device_for_feature, cache_key=cache_key, cache_dir=self.cache_dir)
            if self.feature_dim is None:
                self.feature_dim = feats.shape[1]
                self.cov_n_channels = int(covs.shape[-1])
            elif feats.shape[1] != self.feature_dim:
                raise ValueError(f'Feature dim mismatch: got {feats.shape[1]} for {rec.file_path}, expected {self.feature_dim}')
            elif int(covs.shape[-1]) != int(self.cov_n_channels):
                raise ValueError('Channel count mismatch in cached covariances.')
            n = feats.shape[0]
            feats_all.append(feats)
            covs_all.append(covs)
            labels_all.append(torch.full((n,), int(rec.label), dtype=torch.long))
            subjects_all.append(torch.full((n,), int(rec.subject_id), dtype=torch.long))
            segments_all.append(torch.full((n,), segment_id, dtype=torch.long))
            segment_id += 1
        self.features = torch.cat(feats_all, dim=0)
        self.covs = torch.cat(covs_all, dim=0)
        self.labels = torch.cat(labels_all, dim=0)
        self.subject_ids = torch.cat(subjects_all, dim=0)
        self.segment_ids = torch.cat(segments_all, dim=0)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, idx: int):
        return (self.features[idx], self.labels[idx], self.subject_ids[idx])
