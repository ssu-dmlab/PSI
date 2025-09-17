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
    user_list = pd.read_csv(f"{dataset}/user_list.txt", sep=' ', header=0, dtype={'org_id': str})
    item_list = pd.read_csv(f"{dataset}/item_list.txt", sep=' ', header=0, dtype={'org_id': str})

    valid_users = set(user_list['org_id'])
    valid_items = set(item_list['org_id'])

    # user/item 리스트에 없는 interaction 제거
    raw = raw[raw['user'].isin(valid_users) & raw['item'].isin(valid_items)]

    # user, item, interaction 개수 출력
    print(f"lightGCN user/item list: users={len(valid_users)}, items={len(valid_items)}")
    print(f"preprocessed user/item list: users={raw['user'].nunique()}, items={raw['item'].nunique()}, interactions={len(raw)}")

if __name__ == "__main__":
    '''
    preprocess.py 실행시 생성되는 user, item, interaction 개수 출력
    supported datasets: "amazon-book", "yelp2018"
    yelp2018은 지금 버전이 맞지 않음
    '''
    preprocess("amazon-book")
    # preprocess("yelp2018")