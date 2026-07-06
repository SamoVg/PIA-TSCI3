import kagglehub
import matplotlib
import numpy as np
import pandas as pd
from kagglehub import KaggleDatasetAdapter
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor as PoissonRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "rauffauzanrambe/fifa-world-cup-2026-player-performance-dataset",
    "fifa_world_cup_2026_player_performance.csv",
)
# Columnas a eliminar para evitar ruido, fuga de datos o multicolinealidad
drop_cols = [c for c in [
    # 1. Identificadores y datos logísticos (No aportan valor matemático)
    "player_id", "player_name", "match_id", "match_date", "team", "nationality", 
    "jersey_number", "club_name", "stadium", "city", "opponent_team", "tournament_stage",

    # 2. Fuga de datos (Variables que ocurren después del gol o son consecuencias)
    "match_result", "goals_team", "goals_opponent", 
    "player_rating", "performance_score", "tournament_rating", "player_of_match_awards", 
    "total_goals_tournament", "total_assists_tournament", "total_minutes_tournament",
    "expected_goals_xg", # Se elimina porque es un modelo predictivo en sí mismo que absorberá el peso

    # 3. Métricas de porteros y defensivas irrelevantes (Puros ceros para delanteros)
    "saves", "save_percentage", "punches", "clean_sheet", "goals_conceded", "penalty_saves",

    # 4. Solución a la Multicolinealidad (Eliminamos a los "clones" )
    "successful_passes", "pass_accuracy",  # Dejamos solo 'total_passes'
    "shots_on_target",                     # Dejamos solo 'shots'
    "successful_dribbles",                 # Dejamos solo 'dribbles_attempted'
    "successful_crosses",                  # Dejamos solo 'crosses'
    "aerial_duels_lost",                   # Dejamos solo 'aerial_duels_won'
    "distance_covered_km", "decelerations" # Chocan con 'minutes_played'. Dejamos los sprints/aceleraciones.
] if c in df.columns]

S = df.drop(columns=["goals", *drop_cols])
X = S[df["position"] == "Forward"]
y = df.loc[X.index, "goals"]
assert len(X) == len(y), "X e y deben tener el mismo número de filas."

num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

model = Pipeline(
    steps=[
        (
            "prep",
            ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), num_cols),
                    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
                ]
            ),
        ),
        ("reg", PoissonRegression()),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

feature_names = model.named_steps["prep"].get_feature_names_out()
coef_df = (
    pd.DataFrame(
        {
            "factor": feature_names,
            "coef_estandarizado": model.named_steps["reg"].coef_,
        }
    )
    .assign(impacto_absoluto=lambda d: d["coef_estandarizado"].abs())
    .sort_values("impacto_absoluto", ascending=False)
)

print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")
print("\nTop factores (mayor influencia absoluta en goles):")
print(coef_df[["factor", "coef_estandarizado"]].head(20).to_string(index=False))
assert np.isfinite(r2), "R² inválido: revisa los datos de entrada."

plt.figure(figsize=(10, 5))
top_coef = coef_df.head(10).iloc[::-1]
plt.barh(top_coef["factor"], top_coef["coef_estandarizado"], color="teal")
plt.title("Top 10 factores por coeficiente estandarizado")
plt.xlabel("Coeficiente estandarizado")
plt.ylabel("Factor")
plt.tight_layout()
plt.savefig("top_factores_goles.png", dpi=150)
plt.close()

numeric_df = pd.concat([X.select_dtypes(include=np.number), y.rename("goals")], axis=1)
corr_focus = numeric_df.corr(numeric_only=True)
n = len(corr_focus.columns)
fig_size = min(max(n * 0.55, 11), 32)
plt.figure(figsize=(fig_size, fig_size))
im = plt.imshow(corr_focus, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks(range(len(corr_focus.columns)), corr_focus.columns, rotation=45, ha="right")
plt.yticks(range(len(corr_focus.index)), corr_focus.index)
plt.title("Mapa de correlación: todos los factores numéricos")
plt.tight_layout()
plt.savefig("mapa_correlacion_goles.png", dpi=150)
plt.close()

scoring = df.loc[X.index, ["player_name", "team"]].copy()
scoring["pred_goals"] = model.predict(X).clip(min=0)  # lambda esperado por partido
scoring["prob_meter_gol"] = 1 - np.exp(-scoring["pred_goals"])  # P(goles >= 1), aprox Poisson

ranking = (
    scoring.groupby(["player_name", "team"], as_index=False)
    .agg(
        goles_esperados_prom=("pred_goals", "mean"),
        prob_meter_gol_prom=("prob_meter_gol", "mean"),
        partidos=("pred_goals", "size"),
    )
    .sort_values(["prob_meter_gol_prom", "goles_esperados_prom"], ascending=False)
)

print(ranking.head(20).to_string(index=False))