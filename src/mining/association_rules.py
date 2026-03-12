import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def run_association_rules(df):

    # chỉ lấy một số feature quan trọng
    features = [
        "sex",
        "chest_pain_type",
        "fasting_blood_sugar",
        "exercise_angina",
        "target"
    ]

    basket = df[features]

    # one-hot encode
    basket = pd.get_dummies(basket)

    # convert sang bool (tăng tốc Apriori)
    basket = basket.astype(bool)

    # =====================
    # FREQUENT ITEMSETS
    # =====================

    freq_items = apriori(
        basket,
        min_support=0.15,   # tăng support
        max_len=2,          # giới hạn độ dài itemset
        use_colnames=True
    )

    # =====================
    # ASSOCIATION RULES
    # =====================

    rules = association_rules(
        freq_items,
        metric="confidence",
        min_threshold=0.6
    )

    # lọc rule mạnh
    rules = rules[
        (rules["lift"] > 1.2)
    ]

    # chỉ giữ cột quan trọng
    rules = rules[[
        "antecedents",
        "consequents",
        "support",
        "confidence",
        "lift"
    ]]

    # sort
    rules = rules.sort_values(
        by="lift",
        ascending=False
    )

    # chỉ lấy top rule
    rules = rules.head(30)

    return rules