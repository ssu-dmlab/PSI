import pandas as pd

def preprocess(dataset):
    if dataset == "amazon-book":
        raw = pd.read_csv(f"raw/{dataset}_raw.csv",     
                          names=['user', 'item', 'rating', 'timestamp'],
                          usecols=['user', 'item', 'rating'],
                          header=None,
                          dtype={'user': str, 'item': str, 'rating': float})
    
    elif dataset == "yelp2018":
        raw = pd.read_csv(f"raw/{dataset}_raw.csv", 
                          dtype={'user': str, 'item': str, 'rating': int},
                          header=0)
        
    else:
        raise ValueError("Dataset not supported.")
        
    # 중복행 제거
    raw = raw.drop_duplicates(subset=["user", "item"])

    # user/item mapping list 불러오기
    user_list = pd.read_csv(f"{dataset}/user_list.txt", sep=' ', header=0, dtype={'org_id': str, 'remap_id': int})
    item_list = pd.read_csv(f"{dataset}/item_list.txt", sep=' ', header=0, dtype={'org_id': str, 'remap_id': int})

    valid_users = set(user_list['org_id'])
    valid_items = set(item_list['org_id'])

    # user/item 리스트에 없는 interaction 제거
    raw = raw[raw['user'].isin(valid_users) & raw['item'].isin(valid_items)]

    # org_id → remap_id 매핑
    user_map = dict(zip(user_list['org_id'], user_list['remap_id']))
    item_map = dict(zip(item_list['org_id'], item_list['remap_id']))
    raw['user'] = raw['user'].map(user_map)
    raw['item'] = raw['item'].map(item_map)

    # 원본 데이터 pair에서 raw의 rating 붙이기
    for split in ["train", "test"]:
        origin_path = f"{dataset}/{split}.txt"      # 원본 파일 경로
        out_path = f"{dataset}/{split}_rating.txt"  # 새로운 파일 경로

        with open(origin_path, "r") as f:
            lines = f.readlines()
        
        # user-item pair 생성
        pairs = []
        for line in lines:
            tokens = line.strip().split()
            user_id = int(tokens[0])
            item_ids = [int(x) for x in tokens[1:]]
            for item_id in item_ids:
                pairs.append((user_id, item_id))
        
        # DataFrame 생성
        df_pairs = pd.DataFrame(pairs, columns=['user', 'item'])

        # raw에서 rating 붙이기
        merged = pd.merge(df_pairs, raw[['user', 'item', 'rating']], on=['user', 'item'], how='left')
        
        # 저장
        merged.to_csv(out_path, sep=' ', header=False, index=False)

        # user, item, interaction 개수 출력
        num_users = merged['user'].nunique()
        num_items = merged['item'].nunique()
        num_interactions = len(merged)
        print(f"{split}.txt: users={num_users}, items={num_items}, interactions={num_interactions}")

        # pair 중 raw에 없어서 NaN rating인 경우 확인
        nan_count = merged['rating'].isna().sum()
        print(f"NaN ratings in {split}.txt: {nan_count}")


if __name__ == "__main__":
    '''
    lightGCN의 mapping 파일에 존재하는 pair만 rating 붙여서 저장
    supported datasets: "amazon-book", "yelp2018"
    yelp2018은 지금 버전이 맞지 않음
    '''
    preprocess("amazon-book")
    # preprocess("yelp2018")