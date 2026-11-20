FROM python:3.11-slim-bookworm

COPY --from=eclipse-temurin:21-jre-jammy /opt/java/openjdk /opt/java/openjdk

ENV JAVA_HOME=/opt/java/openjdk
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

WORKDIR /app

COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY run_pipeline.sh .

RUN sed -i 's/\r$//' run_pipeline.sh && chmod +x run_pipeline.sh

CMD ["./run_pipeline.sh"]
