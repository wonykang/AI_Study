import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 사용자-아이템 평점 데이터
data = {
    "Movie1": [5, 3, 0],
    "Movie2": [4, 5, 0],
    "Movie3": [0, 0, 4],
    "Movie4" : [ 0 ,2, 5 ]
}
ratings = pd.DataFrame(data, index=["User1", "User2", "User3"])

# 코사인 유사도 계산
cosine_sim = cosine_similarity(ratings)
cosine_sim_df = pd.DataFrame(cosine_sim, index=ratings.index, columns=ratings.index)

# User1에게 추천
user_similarities = cosine_sim_df.loc["User1"].sort_values(ascending=False)
recommendations = ratings.loc["User2"]  # User2는 User1과 가장 유사
recommendations = recommendations[ratings.loc["User1"] == 0]  # User1이 보지 않은 아이템
print("추천 아이템:\n", recommendations)