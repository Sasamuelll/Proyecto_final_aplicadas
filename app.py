from flask import Flask, render_template, request
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/procesar", methods=["POST"])
def procesar():
    # 1. obtener archivo y opción
    file = request.files["archivo"]
    opcion = request.form["opcion"]
    target = request.form.get("target")  # opcional, para clasificación

    # 2. leer CSV
    df = pd.read_csv(file)

    resultado_texto = ""
    df_resultado = df.copy()

    # 3. aplicar operación
    if opcion == "faltantes":
        # relleno con la media para numéricos y moda para categóricos
        num_cols = df_resultado.select_dtypes(include="number").columns
        cat_cols = df_resultado.select_dtypes(exclude="number").columns

        if len(num_cols) > 0:
            imp_num = SimpleImputer(strategy="mean")
            df_resultado[num_cols] = imp_num.fit_transform(df_resultado[num_cols])

        if len(cat_cols) > 0:
            imp_cat = SimpleImputer(strategy="most_frequent")
            df_resultado[cat_cols] = imp_cat.fit_transform(df_resultado[cat_cols])

        resultado_texto = "Valores faltantes rellenados (media en numéricos, moda en categóricos)."

    elif opcion == "normalizacion":
        num_cols = df_resultado.select_dtypes(include="number").columns
        scaler = MinMaxScaler()
        df_resultado[num_cols] = scaler.fit_transform(df_resultado[num_cols])
        resultado_texto = "Se aplicó normalización Min-Max a las columnas numéricas."

    elif opcion == "discretizacion":
        num_cols = df_resultado.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            disc = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="quantile")
            df_resultado[num_cols] = disc.fit_transform(df_resultado[num_cols])
            resultado_texto = "Se discretizaron las columnas numéricas en 4 bins (cuantiles)."
        else:
            resultado_texto = "No hay columnas numéricas para discretizar."

    elif opcion == "clasificacion":
        if not target or target not in df_resultado.columns:
            resultado_texto = "Debes indicar una columna objetivo válida."
        else:
            X = df_resultado.drop(columns=[target])
            y = df_resultado[target]

            # quitar columnas no numéricas automáticamente
            X = pd.get_dummies(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )

            modelo = DecisionTreeClassifier()
            modelo.fit(X_train, y_train)
            accuracy = modelo.score(X_test, y_test)
            resultado_texto = f"Clasificación con árbol de decisión. Accuracy: {accuracy:.3f}"

    # 4. mandar una muestra de los datos procesados a la plantilla
    preview = df_resultado.head().to_html(classes="table table-striped", index=False)

    return render_template(
        "resultados.html",
        mensaje=resultado_texto,
        tabla_html=preview
    )

if __name__ == "__main__":
    app.run(debug=True)
