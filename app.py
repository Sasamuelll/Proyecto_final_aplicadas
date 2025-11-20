from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)
TEMP_FILE = "resultado_procesado.csv"

# =============================================================
# LIMPIEZA GLOBAL INTELIGENTE
# =============================================================
def limpiar_global(df):
    df = df.astype(str)
    df = df.replace({
        r"\u200b": "", r"\ufeff": "", r"\xa0": "", r"\s+": " "
    }, regex=True)
    df = df.replace({
        "?": np.nan, " ?": np.nan, "? ": np.nan, "  ?": np.nan, " ? ": np.nan,
        "??": np.nan, "???": np.nan
    }, regex=False)
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df = df.applymap(lambda x: x.replace(",", ".") if isinstance(x, str) else x)
    return df

# =============================================================
# DETECCIÓN AUTOMÁTICA DE LA COLUMNA CLASE
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
# IMPUTACIÓN INTELIGENTE
# =============================================================
def imputacion_inteligente(df):
    df = limpiar_global(df)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns.tolist()
    clase_col = detectar_columna_clase(df)
    df_final = df.copy()

    # Caso 1: no hay columna clase → imputación global
    if clase_col is None:
        for col in numeric_cols:
            df_final[col] = pd.to_numeric(df_final[col], errors="coerce")
            df_final[col] = df_final[col].fillna(df_final[col].mean())
        cat_cols = df_final.select_dtypes(include=["object"]).columns.tolist()
        for col in cat_cols:
            moda = df_final[col].dropna().mode()
            df_final[col] = df_final[col].fillna(moda[0] if len(moda) > 0 else "IMPUTADO")
        return df_final

    # Caso 2: imputación por clase
    cat_cols = [c for c in df.columns if c not in numeric_cols and c != clase_col]
    usar_kmeans = True

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

    # Imputación numérica
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
        for col in numeric_cols:
            medias = df_final.groupby(clase_col)[col].mean()
            for clase_val, media_val in medias.items():
                mask = (df_final[col].isna()) & (df_final[clase_col] == clase_val)
                df_final.loc[mask, col] = media_val

    # Imputación categórica por moda de clase
    for col in cat_cols:
        for clase_val in df_final[clase_col].dropna().unique():
            valores = df_final[df_final[clase_col] == clase_val][col].dropna()
            if len(valores) == 0:
                continue
            moda = valores.mode()[0]
            mask = (df_final[col].isna()) & (df_final[clase_col] == clase_val)
            df_final.loc[mask, col] = moda

    # Última imputación global
    for col in numeric_cols:
        df_final[col] = df_final[col].fillna(df_final[col].mean())
    cat_cols_all = df_final.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols_all:
        moda = df_final[col].dropna().mode()
        df_final[col] = df_final[col].fillna(moda[0] if len(moda) > 0 else "IMPUTADO")

    return df_final

# =============================================================
# NORMALIZACIÓN AUTOMÁTICA
# =============================================================
def normalizaciones_generales(df):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns
    if len(numeric_cols) == 0:
        return df
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) < 2:
            continue
        min_val = col_data.min()
        max_val = col_data.max()
        mean_val = col_data.mean()
        std_val = col_data.std()
        df[f"{col}_MinMax"] = (df[col] - min_val) / (max_val - min_val) if max_val != min_val else 0
        df[f"{col}_ZScore"] = (df[col] - mean_val) / std_val if std_val != 0 else 0
        df[f"{col}_Rango"] = 2 * ((df[col] - min_val) / (max_val - min_val)) - 1 if max_val != min_val else 0
        df[f"{col}_Centrado"] = df[col] - mean_val
        k = len(str(int(abs(max_val)))) if max_val != 0 else 1
        df[f"{col}_DecimalScaling"] = df[col] / (10 ** k)
    return df

# =============================================================
# DISCRETIZACIÓN
# =============================================================
def discretizacion(df, metodo="equal_width", bins=5):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) < 2:
            continue
        if metodo == "equal_width":
            df[f"{col}_discretizado"] = pd.cut(df[col], bins=bins, labels=False)
        elif metodo == "equal_freq":
            df[f"{col}_discretizado"] = pd.qcut(df[col], q=bins, labels=False, duplicates="drop")
        else:
            raise ValueError("Método inválido. Use 'equal_width' o 'equal_freq'.")
    return df

# =============================================================
# ARBOL DE DECISION GENERAL
# =============================================================
def arbol_decision_general(df):
    df = imputacion_inteligente(df.copy())
    # Codificar categóricas
    le_dict = {}
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
    objetivo_col = df.columns[-1]
    X = df.drop(objetivo_col, axis=1)
    y = df[objetivo_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier(random_state=42, max_depth=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    tree_rules = export_text(clf, feature_names=list(X.columns))
    return df, acc, report, tree_rules

# =============================================================
# LECTURA DE ARCHIVOS
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
# RUTAS
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

    arbol_info = None

    # ================================
    # PROCESAMIENTO COMPLETO "TODO"
    # ================================
    if accion == "todo":
        mensaje = "Procesamiento completo realizado."

        # 1. Imputación
        df_imputado = imputacion_inteligente(df_original.copy())

        # 2. Normalización
        df_normalizado = normalizaciones_generales(df_imputado.copy())

        # 3. Discretización
        df_discreto = discretizacion(df_normalizado.copy(), metodo="equal_width", bins=5)

        # 4. Árbol de decisión
        _, acc, report, tree_rules = arbol_decision_general(df_original.copy())

        arbol_info = {
            "report": report,
            "rules": tree_rules,
            "accuracy": acc
        }

        # Mandar TODAS las tablas al HTML
        tablas_extra = {
            "imputado": df_imputado.to_html(classes="table table-striped", index=False),
            "normalizado": df_normalizado.to_html(classes="table table-striped", index=False),
            "discreto": df_discreto.to_html(classes="table table-striped", index=False)
        }

        df_resultado = df_discreto.copy()


    # ================================
    # ACCIONES INDIVIDUALES
    # ================================
    elif accion == "imputacion":
        df_resultado = imputacion_inteligente(df_original.copy())
        mensaje = "Imputación realizada correctamente."

    elif accion == "normalizacion":
        df_resultado = normalizaciones_generales(df_original.copy())
        mensaje = "Normalización aplicada correctamente."

    elif accion == "discretizacion":
        df_resultado = discretizacion(df_original.copy(), metodo="equal_width", bins=5)
        mensaje = "Discretización realizada correctamente."

    elif accion == "arbol":
        df_resultado, acc, report, tree_rules = arbol_decision_general(df_original.copy())
        mensaje = f"Árbol entrenado correctamente. Accuracy: {acc:.2f}"
        arbol_info = {"report": report, "rules": tree_rules}

    else:
        df_resultado = df_original.copy()
        mensaje = "Acción inválida."

    # Guardar CSV final
    df_resultado.to_csv(TEMP_FILE, index=False)

    tabla_original = df_original.to_html(classes="table table-striped", index=False)
    tabla_imputada = df_resultado.to_html(classes="table table-striped", index=False)

    return render_template("resultados.html",
                           mensaje=mensaje,
                           tabla_original=tabla_original,
                           tabla_imputada=tabla_imputada,
                           arbol_info=arbol_info,
                           download_link="/descargar")


@app.route("/descargar")
def descargar():
    return send_file(TEMP_FILE, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
