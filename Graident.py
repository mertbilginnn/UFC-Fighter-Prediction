import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

# Verileri yükle ve incele
df = pd.read_csv("C:/Users/mavim/PycharmProjects/UFC_Prediction/fighter_stats.csv")


# Dövüşçüleri siklete göre kategorize et
def categorize_fighter(weight):
    if 52.1 <= weight <= 56.6:
        return "Saman Sıklet"
    elif 56.7 <= weight <= 61.1:
        return "Sinek Sıklet"
    elif 61.2 <= weight <= 65.7:
        return "Horoz Sıklet"
    elif 65.8 <= weight <= 70.2:
        return "Tüy Sıklet"
    elif 70.3 <= weight <= 80.2:
        return "Hafif Sıklet"
    elif 80.3 <= weight <= 89.9:
        return "Orta Sıklet"
    elif 90.0 <= weight <= 110.1:
        return "Hafif Ağır Sıklet"
    else:
        return "Ağır Sıklet"


df["weight_category"] = df["weight"].apply(categorize_fighter)

# Kategorik değişkenleri sayısal değerlere dönüştürme
label_encoder = LabelEncoder()
df["stance"] = label_encoder.fit_transform(df["stance"])

# Her kategori için ayrı model oluştur
for category in df["weight_category"].unique():
    category_data = df[df["weight_category"] == category].copy()  # Veri kopyasını al

    category_data["SApM_inverse"] = 1 / category_data["SApM"]
    category_data["SLpM_inverse"] = 1 / category_data["SLpM"]

    # Kazanma oranını buluyor ve kolon oluşturuyor.
    category_data["win_rate"] = category_data["wins"] / (category_data["wins"] + category_data["losses"])

    X = category_data[
        ["age", "wins", "losses", "height", "weight", "stance", "SLpM", "sig_str_acc", "SApM", "str_def", "td_avg",
         "td_acc", "td_def", "sub_avg"]]
    y = category_data["win_rate"] #Karar Sütunu

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = GradientBoostingRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    print(f"{category} kategorisi için R2 Skor:", model.score(X_test, y_test))

# Yeni bir dövüşçünün özelliklerini içeren bir girdi oluşturun
new_fighter = {
    "age": 48,
    "wins": 20,
    "losses": 5,
    "height": 175,
    "weight": 90,
    "stance": "Orthodox",  # Dövüşçünün dövüş stili (Orthodox, Southpaw, Switch...)
    "SLpM": 1.5,  # Dakika başına gerçekleştirilen önemli saldırılar (significant strikes landed per minute)
    "sig_str_acc": 0.55,  # Önemli saldırıların doğruluğu (significant strike accuracy)
    "SApM": 2.0,  # Dakika başına emilen önemli saldırılar (significant strikes absorbed per minute)
    "str_def": 0.65,  # Rakiplerinin saldırılarından kaçma oranı (strike defense)
    "td_avg": 2.0,  # Dakika başına ortalama takedown (average takedowns per minute)
    "td_acc": 0.6,  # Takedown'ların doğruluğu (takedown accuracy)
    "td_def": 0.8,  # Rakiplerin takedown girişimlerine karşı savunma oranı (takedown defense)
    "sub_avg": 0.5  # Dakika başına ortalama submission
}


# Dövüşçünün siklet kategorisini belirleyin
def categorize_fighter(weight):
    if 52.1 <= weight <= 56.6:
        return "Saman Sıklet"
    elif 56.7 <= weight <= 61.1:
        return "Sinek Sıklet"
    elif 61.2 <= weight <= 65.7:
        return "Horoz Sıklet"
    elif 65.8 <= weight <= 70.2:
        return "Tüy Sıklet"
    elif 70.3 <= weight <= 80.2:
        return "Hafif Sıklet"
    elif 80.3 <= weight <= 89.9:
        return "Orta Sıklet"
    elif 90.0 <= weight <= 110.1:
        return "Hafif Ağır Sıklet"
    else:
        return "Ağır Sıklet"


new_fighter_weight = new_fighter["weight"]
weight_category = categorize_fighter(new_fighter_weight)

# Dövüşçünün özelliklerini içeren bir DataFrame oluşturun
new_fighter_df = pd.DataFrame([new_fighter])

# Yeni dövüşçünün "stance" sütununu One-Hot Encoding ile dönüştür
new_fighter_df = pd.get_dummies(new_fighter_df, columns=["stance"])

new_fighter_df["SApM_inverse"] = 1 / new_fighter_df["SApM"]
new_fighter_df["SLpM_inverse"] = 1 / new_fighter_df["SLpM"]

# Tahmin işlemi için kullanılacak özelliklerin tamamını alın
predict_features = new_fighter_df.reindex(columns=X.columns, fill_value=0)

# Yeni dövüşçünün kazanma olasılığını tahmin edin
new_fighter_win_rate = None  # Tahmin sonucunu saklamak için bir değişken oluşturun
if weight_category in df["weight_category"].unique():
    category_data = df[df["weight_category"] == weight_category].copy()  # Veri kopyasını al
    category_data.loc[:, "win_rate"] = category_data["wins"] / (
            category_data["wins"] + category_data["losses"])  # Kazanma oranını hesapla
    X = category_data[
        ["age", "wins", "losses", "height", "weight", "stance", "SLpM", "sig_str_acc", "SApM", "str_def", "td_avg",
         "td_acc", "td_def", "sub_avg"]]
    y = category_data["win_rate"]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2)  # Yeni satır: Tekrar eğitim öncesi yeni veri ile test edin
    model.fit(X_train, y_train)  # Modeli eğit
    new_fighter_win_rate = model.predict(predict_features)  # Yeni dövüşçünün kazanma olasılığını tahmin et

# Sonuçları yazdırın
if new_fighter_win_rate is not None:
    print(f"{weight_category} kategorisine ait yeni dövüşçünün başarı oranı:", new_fighter_win_rate)
else:
    print("Belirtilen siklet kategorisinde yeterli veri yok veya model eğitimi yapılamadı.")
