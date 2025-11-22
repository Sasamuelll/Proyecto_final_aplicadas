# Usa una imagen base con una versión moderna de Python
FROM python:3.11-slim

# Instala las dependencias de sistema necesarias para scikit-learn
RUN apt-get update -y && \
    apt-get install -y build-essential libopenblas-dev gfortran && \
    rm -rf /var/lib/apt/lists/*

# Establece el directorio de trabajo
WORKDIR /usr/src/app

# Copia e instala las dependencias de Python
COPY requirements.txt .

# ⚠️ CAMBIO: Usamos -v para verbose (más información en el log) para diagnosticar el fallo de pip.
RUN pip install -v --no-cache-dir --disable-pip-version-check -r requirements.txt

# Copia el resto de tu aplicación
COPY . .

# 🚨 CORRECCIÓN: Comando para iniciar el servidor Gunicorn apuntando a app.py
CMD ["gunicorn", "app:app"]
