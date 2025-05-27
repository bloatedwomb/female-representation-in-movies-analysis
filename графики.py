import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("movies.csv")

# Фильмы с 100% женским участием в съемочной группе
films_with_100_percent_women = df[df['crew_female_representation'] == 100]
print("Фильмы с 100% женским составом в съёмочной группе:")
print(films_with_100_percent_women[['title', 'year', 'crew_female_representation']])
print('\n')


# Топ-7 самых популярных фильмов с высокой женской представленностью в съемочной группе
filtered_df = df[df["crew_female_representation"] > 55]
top_movies = filtered_df.sort_values(by="popularity", ascending=False).head(7)
print("Топ-7 самых популярных фильмов с высокой женской представленностью:")
for index, row in top_movies.iterrows():
    print(f"{row['title']} - Популярность: {row['popularity']}, Доля женщин: {row['crew_female_representation']}%")


# График: Средняя доля женщин в съемочной группе по жанру
df = df[df["genres"].notna() & df["crew_female_representation"].notna()]
df["genres"] = df["genres"].apply(eval)
df_exploded = df.explode("genres")
genre_crew_gender = df_exploded.groupby("genres")["crew_female_representation"].mean().sort_values(ascending=False) / 100

plt.figure(figsize=(10, 6))
genre_crew_gender.plot(kind="bar", color=plt.cm.coolwarm(genre_crew_gender.rank(pct=True)))
plt.title("Доля женщин в съёмочной группе по жанрам")
plt.xlabel("Жанр")
plt.ylabel("Средняя доля женщин")
plt.xticks(rotation=45, ha="right")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# График: Доля женщин в кино по годам
df = pd.read_csv("movies.csv")
df = df[df["cast_female_representation"].notna() & df["crew_female_representation"].notna()]
df = df[['year', 'cast_female_representation', 'crew_female_representation']].dropna()

yearly = df.groupby('year')[['cast_female_representation', 'crew_female_representation']].mean()
yearly.replace(0, float('nan'), inplace=True)

plt.figure(figsize=(12, 6))
plt.plot(yearly.index, yearly['cast_female_representation'], label='Актёрский состав', color='lightcoral', marker='o', linestyle='-')
plt.plot(yearly.index, yearly['crew_female_representation'], label='Съёмочная группа', color='violet', marker='o', linestyle='-')

plt.title('Доля женщин в кино по годам')
plt.xlabel('Год')
plt.ylabel('Доля женщин (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# График: Влияние количества женщин в съемочной группе на популярность фильма
df_filtered = df[['popularity', 'crew_female_representation']].dropna()
df_filtered = df_filtered.query("popularity > 0")

plt.figure(figsize=(10, 6))
plt.scatter(df_filtered['crew_female_representation'], df_filtered['popularity'], color='darkcyan', alpha=0.7, edgecolors='k')

plt.xlabel('Доля женщин в съемочной группе')
plt.ylabel('Популярность фильма')
plt.title('Влияние количества женщин в съёмочной группе на популярность фильма')
plt.grid(True)
plt.tight_layout()
plt.show()


# График с ограничением популярности <= 450
df_filtered = df_filtered.query("popularity <= 450")

plt.figure(figsize=(10, 6))
plt.scatter(df_filtered['crew_female_representation'], df_filtered['popularity'], color='darkcyan', alpha=0.7, edgecolors='k')

plt.xlabel('Доля женщин в съемочной группе')
plt.ylabel('Популярность фильма')
plt.title('Влияние количества женщин в съёмочной группе на популярность (с ограничением)')
plt.grid(True)
plt.tight_layout()
plt.show()


# График: Зависимость оценки Бекдел-теста и доли женщин в съёмочной группе
df_filtered = df[['bt_score', 'crew_female_representation']].dropna()
df_filtered['crew_female_representation'] = df_filtered['crew_female_representation'].round()
bechdel_avg = df_filtered.groupby('bt_score')['crew_female_representation'].mean()

plt.figure(figsize=(10, 6))
bechdel_avg.plot(kind='bar', color='darkcyan')

plt.xlabel('Оценка Бекдел-теста')
plt.ylabel('Средняя доля женщин в съемочной группе')
plt.title('Связь оценки Бекдел-теста и доли женщин в съёмочной группе')

plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
