import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('C:/Users/user/Downloads/diabetes.csv')

# Tampilkan informasi dataset
df.info()

# Statistik deskriptif
df.describe()

# Cek nilai 0 pada kolom penting
print("Jumlah nilai 0 pada tiap kolom:")
print((df == 0).sum())

# Ganti nilai 0 dengan median (karena 0 tidak valid untuk kolom berikut)
df['Glucose'] = df['Glucose'].replace(0, df['Glucose'].median())
df['BloodPressure'] = df['BloodPressure'].replace(0, df['BloodPressure'].median())
df['SkinThickness'] = df['SkinThickness'].replace(0, df['SkinThickness'].median())
df['Insulin'] = df['Insulin'].replace(0, df['Insulin'].median())
df['BMI'] = df['BMI'].replace(0, df['BMI'].median())

# Normalisasi fitur
scaler = StandardScaler()
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = scaler.fit_transform(
    df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']]
)

# Visualisasi: Histogram fitur
df.hist(bins=20, figsize=(10, 10))
plt.tight_layout()
plt.show()

# Visualisasi: Distribusi Outcome
sns.countplot(x='Outcome', data=df)
plt.title('Distribusi Outcome (Diabetes vs Tidak Diabetes)')
plt.show()

# Visualisasi: Heatmap Korelasi
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Korelasi Antar Fitur')
plt.show()

# Visualisasi: Boxplot BMI vs Outcome
sns.boxplot(x='Outcome', y='BMI', data=df)
plt.title('Distribusi BMI berdasarkan Outcome')
plt.show()

# Visualisasi: Boxplot Age vs Outcome
sns.boxplot(x='Outcome', y='Age', data=df)
plt.title('Distribusi Usia berdasarkan Outcome')
plt.show()
