from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

app = Flask(__name__)
TEMP_FILE = "resultado_imputado.csv"

# =============================================================
#                 LIMPIEZA GLOBAL INTELIGENTE
# =============================================================
def limpiar_global(df):

    df = df.astype(str)

    df = df.replace({
        r"\u200b": "",
        r"\ufeff": "",
        r"\xa0": "",
        r"\s+": " "
    }, regex=True)

    df = df.replace({
        "?": np.nan,
        " ?": np.nan,
        "? ": np.nan,
        "  ?": np.nan,
        " ? ": np.nan,
        "??": np.nan,
        "???": np.nan
    }, regex=False)

    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df = df.applymap(lambda x: x.replace(",", ".") if isinstance(x, str) else x)

    return df


# =============================================================
#       DETECCIÓN AUTOMÁTICA REAL DE LA COLUMNA CLASE
# =============================================================
def detectar_columna_clase(df):

    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if len(obj_cols) == 0:
        return None
    if len(obj_cols) == 1:
        return obj_cols[0]

    cardinalidades = {col: df[col].nunique() for col in obj_cols}
    clase_col = min(cardinalidades, key=cardinalidades.get)
    return clase_col


# =============================================================
#           IMPUTACIÓN INTELIGENTE COMPLETA
# =============================================================
def imputacion_inteligente(df):

    df = limpiar_global(df)

    # Recuperar tipos numéricos reales
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    numeric_cols = df.select_dtypes(include=["float", "int"]).columns.tolist()
    clase_col = detectar_columna_clase(df)

    df_final = df.copy()

    # =============================================================
    #    CASO 1 — NO HAY COLUMNA CLASE → IMPUTACIÓN GLOBAL
    # =============================================================
    if clase_col is None:

        # Numéricas por media global
        for col in numeric_cols:
            df_final[col] = pd.to_numeric(df_final[col], errors="coerce")
            media = df_final[col].mean()
            df_final[col] = df_final[col].fillna(media)

        # Categóricas por moda global
        cat_cols = df_final.select_dtypes(include=["object"]).columns.tolist()
        for col in cat_cols:
            moda = df_final[col].dropna().mode()
            if len(moda) > 0:
                df_final[col] = df_final[col].fillna(moda[0])
            else:
                df_final[col] = df_final[col].fillna("IMPUTADO")

        return df_final

    # =============================================================
    #    CASO 2 — SÍ HAY CLASE → IMPUTACIÓN POR CLASE
    # =============================================================

    cat_cols = [c for c in df.columns if c not in numeric_cols and c != clase_col]

    usar_kmeans = True

    # Evaluar si KMeans es válido
    if len(numeric_cols) == 0:
        usar_kmeans = False

    for col in numeric_cols:
        if df[col].count() < 4:
            usar_kmeans = False

    for clase in df[clase_col].dropna().unique():
        for col in numeric_cols:
            if df[df[clase_col] == clase][col].count() < 2:
                usar_kmeans = False

    if len(numeric_cols) == 1:
        col = numeric_cols[0]
        if df[col].var() < 1:
            usar_kmeans = False

    # =============================================================
    #       IMPUTACIÓN NUMÉRICA
    # =============================================================
    if usar_kmeans:
        for col in numeric_cols:

            for clase_val in df_final[clase_col].dropna().unique():

                grupo = df_final[df_final[clase_col] == clase_val].copy()
                grupo[col] = pd.to_numeric(grupo[col], errors="coerce")

                temp = grupo[col].fillna(grupo[col].median())

                if len(temp) < 2:
                    continue

                kmeans = KMeans(n_clusters=2, random_state=42)
                clusters = kmeans.fit_predict(temp.values.reshape(-1, 1))

                grupo["cluster"] = clusters
                centros = kmeans.cluster_centers_.flatten()

                for idx in grupo.index:
                    if pd.isna(df_final.loc[idx, col]):
                        df_final.loc[idx, col] = centros[grupo.loc[idx, "cluster"]]

    else:
        # Media por clase
        for col in numeric_cols:
            medias = df_final.groupby(clase_col)[col].mean()
            for clase_val, media_val in medias.items():
                mask = (df_final[col].isna()) & (df_final[clase_col] == clase_val)
                df_final.loc[mask, col] = media_val

    # =============================================================
    #       IMPUTACIÓN CATEGÓRICA POR MODA DE CLASE
    # =============================================================
    for col in cat_cols:
        for clase_val in df_final[clase_col].dropna().unique():

            valores = df_final[df_final[clase_col] == clase_val][col].dropna()

            if len(valores) == 0:
                continue

            moda = valores.mode()[0]

            mask = (df_final[col].isna()) & (df_final[clase_col] == clase_val)
            df_final.loc[mask, col] = moda

    # =============================================================
    #      SI QUEDA ALGÚN NaN → IMPUTACIÓN GLOBAL EXTRA
    # =============================================================

    for col in numeric_cols:
        df_final[col] = df_final[col].fillna(df_final[col].mean())

    cat_cols_all = df_final.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols_all:
        moda = df_final[col].dropna().mode()
        if len(moda) > 0:
            df_final[col] = df_final[col].fillna(moda[0])
        else:
            df_final[col] = df_final[col].fillna("IMPUTADO")

    return df_final


# =============================================================
#                     LECTURA DE ARCHIVOS
# =============================================================
def leer_archivo(file):
    try:
        return pd.read_csv(file, sep=None, engine="python", decimal=",")
    except:
        pass

    try:
        file.seek(0)
        return pd.read_csv(file, sep=";", decimal=",")
    except:
        pass

    try:
        file.seek(0)
        return pd.read_excel(file)
    except:
        pass

    return None

# =============================================================
#     NORMALIZACIÓN AUTOMÁTICA (Min-Max o Z-Score según datos)
# =============================================================

def normalizaciones_generales(df):

    df = df.copy()

    # Detectar columnas numéricas
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns

    if len(numeric_cols) == 0:
        return df  # si no hay nada que normalizar

    for col in numeric_cols:

        col_data = df[col].dropna()

        if len(col_data) < 2:
            continue

        min_val = col_data.min()
        max_val = col_data.max()
        mean_val = col_data.mean()
        std_val = col_data.std()

        # 1. Min-Max
        df[f"{col}_MinMax"] = (df[col] - min_val) / (max_val - min_val) if max_val != min_val else 0

        # 2. Z-Score
        df[f"{col}_ZScore"] = (df[col] - mean_val) / std_val if std_val != 0 else 0

        # 3. Rango [-1, 1]
        df[f"{col}_Rango"] = (
            2 * ((df[col] - min_val) / (max_val - min_val)) - 1
            if max_val != min_val else 0
        )

        # 4. Centrado
        df[f"{col}_Centrado"] = df[col] - mean_val

        # 5. Decimal Scaling
        k = len(str(int(abs(max_val)))) if max_val != 0 else 1
        df[f"{col}_DecimalScaling"] = df[col] / (10 ** k)

    return df



# =============================================================
#                         RUTAS
# =============================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/procesar", methods=["POST"])
def procesar():

    file = request.files["archivo"]
    accion = request.form["accion"]
    df_original = leer_archivo(file)

    if df_original is None:
        return render_template("resultados.html",
            mensaje="No se pudo leer el archivo",
            tabla_original="", tabla_imputada="", download_link=None)

    # SOLO UNA OPCIÓN
    if accion == "imputacion":
        df_resultado = imputacion_inteligente(df_original.copy())
        mensaje = "Imputación realizada correctamente."

    elif accion == "normalizacion":
        df_resultado = normalizaciones_generales(df_original.copy())
        mensaje = "Normalización aplicada a todas las columnas numéricas."



    else:
        mensaje = "Acción inválida."
        df_resultado = df_original.copy()

    df_resultado.to_csv(TEMP_FILE, index=False)

    tabla_original = df_original.to_html(classes="table table-striped", index=False)
    tabla_imputada = df_resultado.to_html(classes="table table-striped", index=False)

    return render_template("resultados.html",
        mensaje=mensaje,
        tabla_original=tabla_original,
        tabla_imputada=tabla_imputada,
        download_link="/descargar")


@app.route("/descargar")
def descargar():
    return send_file(TEMP_FILE, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
