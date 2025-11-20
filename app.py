from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report
import os

app = Flask(__name__)
TEMP_FILE = "resultado_procesado.csv"
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# =============================================================
# VALIDACIÓN DE EXTENSIONES
# =============================================================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# =============================================================
# LIMPIEZA GLOBAL INTELIGENTE
# =============================================================
def limpiar_global(df):
    if df is None or df.empty:
        return df
    
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
    if df is None or df.empty:
        return None
    
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
    if df is None or df.empty:
        return df
    
    if df.shape[0] < 1:
        return df
    
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
            media = df_final[col].mean()
            if pd.notna(media):
                df_final[col] = df_final[col].fillna(media)
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
                
                # Validar que haya varianza
                if len(temp.unique()) < 2:
                    continue
                
                try:
                    kmeans = KMeans(n_clusters=2, random_state=42)
                    clusters = kmeans.fit_predict(temp.values.reshape(-1, 1))
                    grupo["cluster"] = clusters
                    centros = kmeans.cluster_centers_.flatten()
                    for idx in grupo.index:
                        if pd.isna(df_final.loc[idx, col]):
                            df_final.loc[idx, col] = centros[grupo.loc[idx, "cluster"]]
                except Exception:
                    # Si KMeans falla, usar media simple
                    for idx in grupo.index:
                        if pd.isna(df_final.loc[idx, col]):
                            df_final.loc[idx, col] = temp.mean()
                    continue
    else:
        for col in numeric_cols:
            medias = df_final.groupby(clase_col)[col].mean()
            for clase_val, media_val in medias.items():
                if pd.notna(media_val):
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
        media = df_final[col].mean()
        if pd.notna(media):
            df_final[col] = df_final[col].fillna(media)
    cat_cols_all = df_final.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols_all:
        moda = df_final[col].dropna().mode()
        df_final[col] = df_final[col].fillna(moda[0] if len(moda) > 0 else "IMPUTADO")

    return df_final

# =============================================================
# NORMALIZACIÓN AUTOMÁTICA
# =============================================================
def normalizaciones_generales(df):
    if df is None or df.empty:
        return df
    
    df = df.copy()
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns
    if len(numeric_cols) == 0:
        return df
    for col in numeric_cols:
        try:
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
        except Exception as e:
            print(f"Error normalizando columna {col}: {e}")
            continue
    return df

# =============================================================
# DISCRETIZACIÓN
# =============================================================
def discretizacion(df, metodo="equal_width", bins=5):
    if df is None or df.empty:
        return df
    
    df = df.copy()
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns
    for col in numeric_cols:
        try:
            col_data = df[col].dropna()
            if len(col_data) < 2:
                continue
            if metodo == "equal_width":
                df[f"{col}_discretizado"] = pd.cut(df[col], bins=bins, labels=False)
            elif metodo == "equal_freq":
                df[f"{col}_discretizado"] = pd.qcut(df[col], q=bins, labels=False, duplicates="drop")
            else:
                raise ValueError("Método inválido. Use 'equal_width' o 'equal_freq'.")
        except Exception as e:
            print(f"Error discretizando columna {col}: {e}")
            continue
    return df

# =============================================================
# ARBOL DE DECISION GENERAL
# =============================================================
def arbol_decision_general(df):
    if df is None or df.empty:
        return df, 0, "No hay datos para entrenar", ""
    
    if df.shape[0] < 10:
        return df, 0, "Datos insuficientes para entrenar el modelo (mínimo 10 filas)", ""
    
    if df.shape[1] < 2:
        return df, 0, "Se necesitan al menos 2 columnas (features + objetivo)", ""
    
    df = imputacion_inteligente(df.copy())
    
    if df.empty:
        return df, 0, "Error en el procesamiento de datos", ""
    
    # Codificar categóricas
    le_dict = {}
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
    objetivo_col = df.columns[-1]
    X = df.drop(objetivo_col, axis=1)
    y = df[objetivo_col]
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = DecisionTreeClassifier(random_state=42, max_depth=5)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        tree_rules = export_text(clf, feature_names=list(X.columns))
        return df, acc, report, tree_rules
    except Exception as e:
        return df, 0, f"Error al entrenar el modelo: {str(e)}", ""

# =============================================================
# LECTURA DE ARCHIVOS
# =============================================================
def leer_archivo(file):
    if file is None or file.filename == '':
        return None
    
    # Validar tamaño del archivo
    file.seek(0, 2)  # Ir al final
    size = file.tell()
    file.seek(0)  # Volver al inicio
    
    if size == 0:
        return None
    
    if size > MAX_FILE_SIZE:
        return None
    
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
    ACCIONES_VALIDAS = ['imputacion', 'normalizacion', 'discretizacion', 'arbol', 'todo']
    
    # Validar que el archivo existe
    if 'archivo' not in request.files:
        return render_template("resultados.html",
                               mensaje="No se ha enviado ningún archivo",
                               tabla_original="", tabla_imputada="", download_link=None)
    
    file = request.files["archivo"]
    
    # Validar archivo vacío
    if not file or file.filename == '':
        return render_template("resultados.html",
                               mensaje="No se ha seleccionado ningún archivo",
                               tabla_original="", tabla_imputada="", download_link=None)
    
    # Validar extensión
    if not allowed_file(file.filename):
        return render_template("resultados.html",
                               mensaje="Formato de archivo no válido. Use CSV o Excel (.xlsx, .xls)",
                               tabla_original="", tabla_imputada="", download_link=None)
    
    # Validar acción
    accion = request.form.get("accion", "")
    if accion not in ACCIONES_VALIDAS:
        return render_template("resultados.html",
                               mensaje="Acción no válida",
                               tabla_original="", tabla_imputada="", download_link=None)

    df_original = leer_archivo(file)
    
    # Validar lectura correcta
    if df_original is None:
        return render_template("resultados.html",
                               mensaje="No se pudo leer el archivo. Verifique que sea un CSV o Excel válido y que no exceda 50MB",
                               tabla_original="", tabla_imputada="", download_link=None)
    
    # Validar que no esté vacío
    if df_original.empty or df_original.shape[0] == 0 or df_original.shape[1] == 0:
        return render_template("resultados.html",
                               mensaje="El archivo no contiene datos válidos",
                               tabla_original="", tabla_imputada="", download_link=None)

    arbol_info = None
    tablas_extra = None

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
        arbol_info = {"report": report, "rules": tree_rules, "accuracy": acc}

    else:
        df_resultado = df_original.copy()
        mensaje = "Acción inválida."

    # Validar que el resultado no esté vacío
    if df_resultado is None or df_resultado.empty:
        return render_template("resultados.html",
                               mensaje="Error: el procesamiento no generó resultados válidos",
                               tabla_original=df_original.to_html(classes="table table-striped", index=False),
                               tabla_imputada="", download_link=None)

    # Guardar CSV final
    try:
        df_resultado.to_csv(TEMP_FILE, index=False)
    except Exception as e:
        tabla_original = df_original.to_html(classes="table table-striped", index=False)
        tabla_imputada = df_resultado.to_html(classes="table table-striped", index=False)
        return render_template("resultados.html",
                               mensaje=f"Error al guardar el archivo: {str(e)}",
                               tabla_original=tabla_original,
                               tabla_imputada=tabla_imputada,
                               arbol_info=arbol_info,
                               tablas_extra=tablas_extra,
                               download_link=None)

    tabla_original = df_original.to_html(classes="table table-striped", index=False)
    tabla_imputada = df_resultado.to_html(classes="table table-striped", index=False)

    return render_template("resultados.html",
                           mensaje=mensaje,
                           tabla_original=tabla_original,
                           tabla_imputada=tabla_imputada,
                           arbol_info=arbol_info,
                           tablas_extra=tablas_extra,
                           download_link="/descargar")


@app.route("/descargar")
def descargar():
    # Validar que el archivo existe
    if not os.path.exists(TEMP_FILE):
        return "Archivo no encontrado. Por favor, procese los datos primero.", 404
    
    try:
        return send_file(TEMP_FILE, as_attachment=True)
    except Exception as e:
        return f"Error al descargar el archivo: {str(e)}", 500

if __name__ == "__main__":
    app.run()