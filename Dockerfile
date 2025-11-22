# Usa una imagen base con una versión moderna de Python
FROM python:3.11-slim

# Instala las dependencias de sistema necesarias para scikit-learn
RUN apt-get update -y && \
    # CAMBIO AQUÍ: libopenblas-dev en lugar de libatlas-base-dev
    apt-get install -y build-essential libopenblas-dev gfortran && \
    rm -rf /var/lib/apt/lists/*

# Establece el directorio de trabajo
WORKDIR /usr/src/app

# Copia e instala las dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto de tu aplicación
COPY . .

# Comando para iniciar el servidor Gunicorn
CMD ["gunicorn", "app:app"]
