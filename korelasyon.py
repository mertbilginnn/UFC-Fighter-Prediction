import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Veri setini yükleme
df = pd.read_csv("C:/Users/mavim/PycharmProjects/UFC_Prediction/fighter_stats.csv")



# Belirtilen özelliklerin kazanma durumu ile korelasyon analizi
ozellikler = ["SLpM", "sig_str_acc", "SApM",
              "str_def", "td_avg", "td_acc",
              "td_def", "sub_avg", 'age']

kazanma_durumu_korelasyon = df[ozellikler].corrwith(df['wins'])

# Korelasyonları bir veri çerçevesine yerleştirme
korelasyon_df = pd.DataFrame(kazanma_durumu_korelasyon, columns=['Kazanma Durumu ile Korelasyon'])

# Korelasyon değerlerini sıralama
korelasyon_df = korelasyon_df.sort_values(by='Kazanma Durumu ile Korelasyon', ascending=False)

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.barh(korelasyon_df.index, korelasyon_df['Kazanma Durumu ile Korelasyon'], color='skyblue')
plt.xlabel('Korelasyon')
plt.title('Kazanma Durumu ile Özellikler Arasındaki Korelasyon')
plt.gca().invert_yaxis()  # Grafikte en yüksek korelasyon en üstte olacak şekilde sıralama
plt.show()


# Kategorik sütunu kazanma durumu ile gruplama ve ortalama kazanma durumu değerlerini hesaplama
kategorik_sutun = 'stance'  # Burada kendi kategorik sütununuzun adını vermelisiniz
sns.barplot(x=kategorik_sutun, y='wins', data=df, estimator=lambda x: sum(x)/len(x)*1)
plt.title('Duruşun Kazanmaya Olan Etkisi')
plt.xlabel('Open Stance  -  Orthodox  -   Southpaw    -  Switch   -    Sideways')
plt.ylabel('Kazanılan Ortalama Maç')
plt.show()


# Veri setini yükleyin
df = pd.read_csv("C:/Users/mavim/PycharmProjects/UFC_Prediction/fighter_stats.csv")


