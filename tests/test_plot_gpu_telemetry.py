from __future__ import annotations

import json

from scripts.plot_gpu_telemetry import (
    load_telemetry_rows,
    plot_util_vs_time,
    plot_vs_concurrency,
    write_summary_md,
)


def _write_cell(path, framework, mode, conc, throughput, util_mean, *, csv_path=None):
    telem = {
        "util_gpu_mean": util_mean,
        "util_gpu_p95": min(100.0, util_mean + 5),
        "mem_used_mib_max": 10000.0,
        "n_samples": 20,
    }
    if csv_path is not None:
        telem["csv_path"] = str(csv_path)
    path.write_text(
        json.dumps(
            {
                "cell": f"{framework}_c{conc}_{mode}",
                "config": {
                    "framework": framework,
                    "model": "openai/whisper-large-v3-turbo",
                    "device": "cuda",
                    "concurrency": conc,
                    "dispatch_mode": mode,
                },
                "env": {"gpu": "Test GPU"},
                "summary": {"throughput_audio_s_per_wall_s": throughput},
                "gpu_telemetry": telem,
            }
        )
    )


def test_plot_vs_concurrency(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    out = tmp_path / "published"
    for fw, mode, conc, thr, util in [
        ("hf_transformers", "serialized", 1, 32.8, 40.0),
        ("hf_transformers", "serialized", 8, 32.7, 45.0),
        ("vllm", "concurrent", 8, 242.4, 95.0),
        ("vllm", "concurrent", 32, 255.2, 97.0),
    ]:
        _write_cell(results / f"{fw}_c{conc}_{mode}_latest.json", fw, mode, conc, thr, util)

    rows = load_telemetry_rows(results)
    png = out / "gpu_util_vs_concurrency.png"
    plot_vs_concurrency(rows, png)
    write_summary_md(rows, out / "gpu_util_vs_concurrency.md", duty_cycle=False)
    assert png.exists() and png.stat().st_size > 0


def test_plot_util_vs_time_c8(tmp_path):
    results = tmp_path / "results"
    telem = results / "gpu_telemetry"
    telem.mkdir(parents=True)
    hf_csv = telem / "hf.csv"
    vllm_csv = telem / "vllm.csv"
    hf_csv.write_text(
        "timestamp,util_gpu,util_mem,mem_used_mib,power_w\n"
        + "\n".join(f"t{i},{u},10,8000,150" for i, u in enumerate([20, 80, 15, 85]))
        + "\n"
    )
    vllm_csv.write_text(
        "timestamp,util_gpu,util_mem,mem_used_mib,power_w\n"
        + "\n".join(f"t{i},{u},10,8000,150" for i, u in enumerate([90, 92, 95, 93]))
        + "\n"
    )
    _write_cell(
        results / "hf_latest.json", "hf_transformers", "serialized", 8, 32.7, 50.0, csv_path=hf_csv
    )
    _write_cell(
        results / "vllm_latest.json", "vllm", "concurrent", 8, 242.4, 94.0, csv_path=vllm_csv
    )
    rows = load_telemetry_rows(results)
    duty = tmp_path / "published" / "gpu_util_vs_time_c8.png"
    assert plot_util_vs_time(rows, duty) is True
    assert duty.exists()
