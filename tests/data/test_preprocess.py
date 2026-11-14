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
