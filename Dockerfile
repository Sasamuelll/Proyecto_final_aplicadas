# Usa una imagen base con una versión moderna de Python
FROM python:3.11-slim

# Instala las dependencias de sistema necesarias para scikit-learn
RUN apt-get update -y && \
    apt-get install -y build-essential libatlas-base-dev gfortran && \
    rm -rf /var/lib/apt/lists/*

# Establece el directorio de trabajo
WORKDIR /usr/src/app

# Copia e instala las dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto de tu aplicación
COPY . .

# Comando para iniciar el servidor Gunicorn
CMD ["gunicorn", "tu_archivo_app:app"] 
# ⚠️ CAMBIA 'tu_archivo_app:app' por el nombre de tu archivo principal y la variable de Flask.
