import numpy as np
import pandas as pd

from seqrec.data.preprocess import build_artifacts


def test_filter_index_split_and_train_only_popularity():
    rows = []
    for user in (1, 2, 3, 4):
        for t, item in enumerate((40, 10, 30, 20)):  # file order differs from time order below
            rows.append((user, item, 100 - t if user == 1 else t))
    rows.append((1, 99, 500))  # item 99 has a single user -> dropped
    rows.append((5, 10, 1))  # user 5 has a single interaction -> dropped
    df = pd.DataFrame(rows, columns=["user_id", "item_id", "timestamp"])

    art = build_artifacts(df, min_user_count=3, min_item_count=2)

    assert art.num_users == 4 and art.num_items == 4
    assert art.model_to_orig.tolist() == [0, 10, 20, 30, 40]
    user_items = [art.items[art.offsets[u] : art.offsets[u + 1]].tolist() for u in range(4)]
    assert user_items[1] == [4, 1, 3, 2]  # ordered by timestamp (40, 10, 30, 20)
    assert user_items[0] == [2, 3, 1, 4]  # user 1 has reversed timestamps
    assert art.split[art.offsets[1] : art.offsets[2]].tolist() == [0, 0, 1, 2]
    # popularity counts only the two train items of each user
    assert art.popularity[0] == 0
    assert art.popularity.sum() == 8
    assert np.array_equal(art.popularity, np.bincount(art.items[art.split == 0], minlength=5))


def test_zvuk_reader_keeps_first_real_listens_of_popular_tracks(tmp_path):
    import pandas as pd

    from seqrec.data.preprocess import read_zvuk

    t = pd.Timestamp("2023-03-01")
    rows = [  # user, track, seconds after t, play duration
        (1, 10, 0, 100.0), (1, 10, 5, 100.0),  # repeat: only the first listen counts
        (1, 11, 1, 2.0),                       # a skip
        (1, 12, 2, 50.0), (2, 12, 3, 60.0), (3, 12, 4, 31.0),
        (2, 10, 6, 40.0), (3, 13, 7, 45.0),
    ]
    frame = pd.DataFrame({
        "user_id": [r[0] for r in rows], "session_id": 0,
        "datetime": [t + pd.Timedelta(seconds=r[2]) for r in rows],
        "track_id": [r[1] for r in rows], "play_duration": [r[3] for r in rows],
    }).astype({"user_id": "int32", "session_id": "int32", "track_id": "int32", "play_duration": "float32"})
    frame["datetime"] = frame["datetime"].astype("datetime64[ms]")
    path = tmp_path / "zvuk.parquet"
    frame.to_parquet(path)

    df = read_zvuk(path, min_play_seconds=30)
    assert sorted(map(tuple, df[["user_id", "item_id"]].to_numpy())) == [(1, 10), (1, 12), (2, 10), (2, 12), (3, 12), (3, 13)]
    first = df[(df.user_id == 1) & (df.item_id == 10)].timestamp.item()
    assert first == int(t.value // 1_000_000)  # milliseconds of the first listen
    capped = read_zvuk(path, min_play_seconds=30, max_items=2)  # tracks 12 (3 listeners) and 10 (2)
    assert set(capped.item_id) == {10, 12}
