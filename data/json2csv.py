import pandas as pd

# yelp 2018 dataset 처리에 사용됨
# JSON 파일 읽기
reviews = pd.read_json("./raw/yelp_academic_dataset_review.json", lines=True)

# 필요한 컬럼만 선택 및 이름 변경
reviews = reviews[["user_id", "business_id", "stars"]]
reviews = reviews.rename(columns={"user_id": "user", "business_id": "item", "stars": "rating"})

# CSV 파일로 저장
reviews.to_csv("./raw/yelp2018_raw.csv", index=False)