from pathlib import Path
import os
from types import SimpleNamespace
import importlib.util


def load_run_train():
    path = Path("improved-diffusion/scripts/run_train.py")
    spec = importlib.util.spec_from_file_location("run_train", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_build_command_uses_project_and_work_dirs() -> None:
    run_train = load_run_train()
    project_root, work_dir, script_dir = run_train.project_paths()

    args = SimpleNamespace(
        experiment="random",
        model_arch="transformer",
        modality="e2e-tgt",
        noise_schedule="sqrt",
        loss_type="Lsimple",
        dropout="0.1",
        weight_decay=0.0,
        image_size=8,
        hidden_size=128,
        in_channel=16,
        m=3,
        k=32,
        lr_anneal_steps=200000,
        num_res_blocks=2,
        lr=0.0001,
        bsz=64,
        diff_steps=2000,
        padding_mode="block",
        seed=102,
        notes="xstart_e2e",
        app="--predict_xstart True --training_mode e2e --vocab_size 821 --e2e_train ../datasets/e2e_data",
    )

    model_file = run_train.build_model_file(args, project_root)
    cmd = run_train.build_command(args, model_file, work_dir, script_dir)
    expected_fork_safe = os.environ.get("RDMAV_FORK_SAFE", "1")

    assert f"cd {work_dir}" in cmd
    assert f"RDMAV_FORK_SAFE={expected_fork_safe}" in cmd
    assert str(Path(project_root) / "diffusion_models") in model_file
    assert str(Path(script_dir) / "train.py") in cmd
    assert f"OPENAI_LOGDIR={model_file}" in cmd
