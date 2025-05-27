import pandas as pd
import ast
from collections import Counter
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv("movies.csv")

# Подсчет количества слов в названиях фильмов, которые начинаются на "S"
def count_words_starting_with_s(title):
    words = title.split()
    return sum(1 for word in words if word.startswith('S'))

word_count = df['title'].dropna().apply(count_words_starting_with_s)
total_words = word_count.sum()

print(f"Общее количество слов в названиях, начинающихся на 'S': {total_words}\n")


# Определение двух наиболее часто встречающихся жанровых пар
genre_pairs = []

for i in df['genres']:
    try:
        genres = ast.literal_eval(i) if pd.notna(i) else []
        if isinstance(genres, list) and len(genres) >= 2:
            genre_pairs.extend(combinations(sorted(genres), 2))
    except (ValueError, SyntaxError):
        continue

most_common = Counter(genre_pairs).most_common(5)

print("Пять самых популярных пар жанров:")
for pair, count in most_common:
    print(f"{pair} - {count} раз")
print("\n")


# Фильтрация фильмов с bt_score=3, выпущенных после 2000 года, в жанре Fantasy
def has_fantasy(genres_str):
    try:
        genres = ast.literal_eval(genres_str)
        return 'Fantasy' in genres
    except:
        return False

filtered = df[
    (df['bt_score'] == 3) &
    (df['year'] > 2000) &
    (df['genres'].apply(has_fantasy))
]

print("Фильмы с bt_score=3, с датой выпуска после 2000 года в жанре Fantasy:")
print(filtered[['title', 'year', 'bt_score', 'genres']], "\n")


# Построение диаграммы по среднему количеству мужчин и женщин в фильмах с одним жанром
df_one_genre = df[df['genres'].apply(lambda g: isinstance(ast.literal_eval(g), list) and len(ast.literal_eval(g)) == 1)].copy()
df_one_genre['genre'] = df_one_genre['genres'].apply(lambda g: ast.literal_eval(g)[0])

num_movies = len(df_one_genre)
print(f"Количество фильмов с одним жанром: {num_movies}")

df_one_genre['male_count'] = df_one_genre['cast_gender'].apply(lambda cg: ast.literal_eval(cg)[0] if pd.notna(cg) else 0)
df_one_genre['female_count'] = df_one_genre['cast_gender'].apply(lambda cg: ast.literal_eval(cg)[1] if pd.notna(cg) else 0)

grouped = df_one_genre.groupby('genre')[['male_count', 'female_count']].mean()

grouped.plot(
    kind='bar',
    figsize=(10, 6),
    color=['cornflowerblue', 'crimson']
)
plt.title('Среднее количество мужчин и женщин в фильмах с одним жанром')
plt.xlabel('Жанр')
plt.ylabel('Среднее количество')
plt.xticks(rotation=45)
plt.legend(['Мужчины', 'Женщины'])
plt.tight_layout()
plt.show()


# Прогноз % количества женщин в cast_gender до 2050 года
data = df[['year', 'cast_female_representation']].dropna()
data = data[data['year'] > 0]
data_grouped = data.groupby('year').mean().reset_index()

X = data_grouped[['year']]
y = data_grouped['cast_female_representation']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
print(f"R^2 (коэффициент детерминации) модели: {r2:.3f}")

years_future = pd.DataFrame({'year': range(data_grouped['year'].max() + 1, 2051)})
forecast = model.predict(years_future)

plt.figure(figsize=(10,6))
plt.scatter(data_grouped['year'], y, alpha=0.5)
plt.plot(X_test['year'], y_pred, 'r.')
plt.plot(years_future['year'], forecast, 'g-')
plt.show()
