import pandas as pd
import matplotlib.pyplot as plt



# CSV dosyasından veriyi oku
df = pd.read_csv("C:/Users/mavim/PycharmProjects/UFC_Prediction/fighter_stats.csv")

# Dövüşçülerin sınıflandırılması için kilo aralıkları
siniflar = {
    "Saman Sıklet": (52.1, 56.6),
    "Sinek Sıklet": (56.7, 61.1),
    "Horoz Sıklet": (61.2, 65.7),
    "Tüy Sıklet": (65.8, 70.2),
    "Hafif Sıklet": (70.3, 80.2),
    "Orta Sıklet": (80.3, 89.9),
    "Hafif Ağır Sıklet": (90.0, 110.1),
    "Ağır Sıklet": (110.2, 350)
}

# Dövüşçüleri sınıflandırma fonksiyonu
def siniflandir(weight):
    for sinif, (alt_sinir, ust_sinir) in siniflar.items():
        if alt_sinir <= weight <= ust_sinir:
            return sinif
    return "Belirsiz"

# Veri setindeki kilo sütununu kullanarak dövüşçüleri sınıflandırma
df['Sınıf'] = df['weight'].apply(siniflandir)

# Sınıf dağılımını görselleştirme
sınıf_sayilari = df['Sınıf'].value_counts()

plt.figure(figsize=(10, 6))  # Grafik boyutunu genişlettik
plt.bar(sınıf_sayilari.index, sınıf_sayilari.values, color='blue')
plt.xlabel('Sınıflar')
plt.ylabel('Dövüşçü Sayısı')
plt.title('Dövüşçülerin Sınıf Dağılımı')
plt.xticks(rotation=45, fontsize=8)  # Yazı boyutunu ve konumunu ayarladık
plt.tight_layout()  # Grafik düzenini iyileştirme
plt.xticks(rotation=45)
plt.show()


