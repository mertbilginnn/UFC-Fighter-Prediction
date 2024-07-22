import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Dosyayı oku
df = pd.read_csv("C:/Users/mavim/PycharmProjects/UFC_Prediction/fighter_stats.csv")

# Stance sütununu kodlama yapısına dönüştür
stance_dict = {
    'Open Stance': 0,
    'Orthodox': 1,
    'Southpaw': 2,
    'Switch': 3,
    'Sideways': 4
}
df['stance'] = df['stance'].replace(stance_dict)

# Gelecekteki pandas sürümlerindeki değişikliklere hazırlık yap zorunlu istiyor
pd.set_option('future.no_silent_downcasting', True)

# Label encoding uygula
label_encoder = LabelEncoder()
df['Dövüşçü Kimliği'] = label_encoder.fit_transform(df['name'])

# Eksik değerleri işle
df['stance'] = df['stance'].fillna(df['stance'].mode()[0])
df['age'] = df['age'].fillna(df['age'].median())
df['reach'] = df['reach'].fillna(df['reach'].mean())

# Eksik değerleri yazdır
print(df.isnull().sum())

# Sayısal sütunlardaki değerlerin formatını düzenle
df['reach'] = df['reach'].round(2)
df['weight'] = df['weight'].round(1)
df['age'] = df['age'].round(1)
df['SLpM'] = df['SLpM'].round(2)
df['sig_str_acc'] = df['sig_str_acc'].round(2)
df['SApM'] = df['SApM'].round(2)
df['str_def'] = df['str_def'].round(2)
df['td_avg'] = df['td_avg'].round(2)
df['td_acc'] = df['td_acc'].round(2)
df['td_def'] = df['td_def'].round(2)
df['sub_avg'] = df['sub_avg'].round(2)

# Temizlenmiş veriyi kaydet
df.to_csv("fighter_stats.csv", index=False)
