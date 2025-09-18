from sklearn.preprocessing import OneHotEncoder
import pandas as pd

encoder = OneHotEncoder(handle_unknown="ignore")

# alphabet = [chr(i) for i in range(ord("a"), ord("z") + 1)]

# X_cat = []

# for letter in alphabet:
#     X_cat.append([letter])
# encoder.fit(X_cat)

X_cat = [["Java"], ["Python"], ["C++"], ["Java"], ["C#"], ["Python"]]

encoder.fit(X_cat)

print(encoder.transform([["Python"]]).toarray())


# Ví dụ dữ liệu
df = pd.DataFrame({
    "Suburb": ["Richmond", "Carlton", "Abbotsford", "Richmond", "Richmond", "Carlton", "Docklands"],
    "Rooms": [3, 2, 4, 3, 5, 2, 1],
    "Price": [1200_000, 950_000, 1_450_000, 1_250_000, 1_800_000, 1_000_000, 650_000]
})

# Frequency Encoding: thay Suburb bằng tần suất xuất hiện
# hoặc .value_counts(normalize=True) nếu muốn tần suất %
freq_map = df["Suburb"].value_counts()
df["Suburb_freq"] = df["Suburb"].map(freq_map)

print(df[["Suburb", "Suburb_freq"]])
