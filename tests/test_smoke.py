from pathlib import Path


def test_e2e_dataset_present() -> None:
    assert Path("datasets/e2e_data/src1_train.txt").exists()


def test_improved_diffusion_importable() -> None:
    import improved_diffusion  # noqa: F401
