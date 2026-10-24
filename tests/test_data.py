from pathlib import Path

import pandas as pd
import pytest

from immcan.data import CandidateRankingDataset, make_loader


def example_row():
    return {
        "compound": "break the ice",
        "sentence": "They broke the ice.",
        "sentence_type": "idiomatic",
        "expected_order": str(["2.png", "4.png", "1.png", "5.png", "3.png"]),
        **{f"image{i}_name": f"{i}.png" for i in range(1, 6)},
        **{f"image{i}_caption": f"Caption {i}" for i in range(1, 6)},
    }


def test_rank_targets_follow_image_columns_not_answer_order(tmp_path: Path):
    path = tmp_path / "data.tsv"
    pd.DataFrame([example_row()]).to_csv(path, sep="\t", index=False)
    sample = CandidateRankingDataset(path, tmp_path, load_images=False)[0]
    assert sample["names"] == ["1.png", "2.png", "3.png", "4.png", "5.png"]
    assert sample["ranks"].tolist() == [2, 0, 4, 1, 3]
    assert sample["relevance"].tolist() == [0.5, 1.0, 0.0, 0.75, 0.25]


def test_incorrect_root_count_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="equal"):
        make_loader(["a.tsv", "b.tsv"], ["root"], load_images=False,
                    batch_size=1, shuffle=False)
