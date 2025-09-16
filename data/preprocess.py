import pandas as pd
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import os

def preprocess(dataset, K):
    raw = pd.read_csv(f"raw/{dataset}_raw.csv", dtype={'user': str, 'item': str, 'rating': float})
    ## 중복행 제거
    raw = raw.drop_duplicates(subset=["user", "item"])
    ## 정수형 인코딩으로 변경
    if dataset != "ML1M":
        raw["user"] = raw["user"].factorize()[0]
        raw["item"] = raw["item"].factorize()[0]
    ## K-core 필터링
    prev_pairs = -1
    iteration = 0
    kept_users = set(raw["user"].unique())
    kept_items = set(raw["item"].unique())

    print(f"Dataset: {dataset}, K: {K}")
    print(f"Initial users: {len(kept_users)}, items: {len(kept_items)}")
    while True:
        iteration += 1
        # 유저 차수 필터
        user_deg = raw.groupby("user")["item"].nunique()
        kept_users = set(user_deg[user_deg >= K].index)
        raw = raw[raw["user"].isin(kept_users)]

        # 아이템 차수 필터
        item_deg = raw.groupby("item")["user"].nunique()
        kept_items = set(item_deg[item_deg >= K].index)
        raw = raw[raw["item"].isin(kept_items)]

        cur_pairs = len(raw)
        print(f"Iter {iteration}: users={len(kept_users)}, items={len(kept_items)}, pairs={cur_pairs}")
        if cur_pairs == prev_pairs:
            break
        prev_pairs = cur_pairs

        if cur_pairs == 0:
            break

    # 원본 데이터에서 최종 유저/아이템만 남김
    seed = 42
    np.random.seed(seed)
    
    raw = raw[raw["user"].isin(kept_users) & raw["item"].isin(kept_items)]

    ### 라벨링 및 데이터셋 분할
    threshold = 4.0
    raw.rename(columns={"rating": "label"}, inplace=True)

    ## 라벨을 {1, 0}으로 변경
    raw["label"] = (raw["label"] >= 4.0).astype(int)
    rng = np.random.default_rng(seed)



    # 아이템 글로벌 차수(LOO 시 train에 남기는 우선순위 판단에 사용)
    global_item_deg = raw["item"].value_counts()

    # 유저별 그룹
    groups = raw.groupby("user", sort=False)

    test_idx = []
    skipped_users = 0
    used_users = 0

    for u, df_u in groups:
        # 유저 차수 1이면 LOO 불가 → 전부 train
        if len(df_u) <= 1:
            skipped_users += 1
            continue

        # 후보 선택 우선순위:
        # (a) positive & item_deg>1
        # (b) positive
        # (c) any & item_deg>1
        # (d) any
        def pick_index(candidates):
            if len(candidates) == 0:
                return None
            # 랜덤
            return rng.choice(df_u.loc[candidates].index, size=1)[0]

        # boolean masks
        pos_mask = (df_u["label"] == 1)
        deg_gt1_mask = df_u["item"].map(lambda it: global_item_deg.get(it, 0) > 1)

        # 우선순위별 후보 인덱스
        cand_a = df_u.index[pos_mask & deg_gt1_mask]
        cand_b = df_u.index[pos_mask]
        cand_c = df_u.index[deg_gt1_mask]
        cand_d = df_u.index

        chosen = (pick_index(cand_a) or
                  pick_index(cand_b) or
                  pick_index(cand_c) or
                  pick_index(cand_d))

        test_idx.append(chosen)
        used_users += 1

        # 아이템 차수를 즉시 업데이트해 (이후 유저 판단 시 좀 더 정확)
        it = df_u.loc[chosen, "item"]
        global_item_deg[it] = max(global_item_deg[it] - 1, 0)

    test = raw.loc[test_idx].copy()
    train = raw.drop(index=test_idx).copy()

    
    # (선택) test 쪽 아이템이 train에 전혀 없게 된 edge가 있다면 로깅
    cold_items_in_train = set(test["item"].unique()) - set(train["item"].unique())
    if len(cold_items_in_train) > 0:
        print(f"[Warn] {len(cold_items_in_train)} items appear only in test (not in train). "
              f"Consider K>=2 or adjust selection if this is undesirable.")

    print(f"Users with LOO applied: {used_users}, users skipped (deg=1): {skipped_users}")
    print(f"Final: train={len(train)}, test={len(test)}")

    os.makedirs(f"processed/{dataset}", exist_ok=True)
    train.to_csv(f"processed/{dataset}_{K}core_train.csv", index=False)
    test.to_csv(f"processed/{dataset}_{K}core_test.csv", index=False)

if __name__ == "__main__":
    preprocess("ML1M", 10)
    preprocess("Books", 10)
    preprocess("Yelp", 10)