# Plan de milestones — Proyecto de predicción de churn (ecommerce)

*Complementa a `contexto-pivote-negocio-ecommerce.md`. Este documento es el desglose operativo del Proyecto A mencionado ahí: predicción de churn para ecommerce, doble propósito (portafolio ML + prueba de concepto del producto Órbita).*

Arquitectura de referencia (ya acordada, no se repite el detalle completo acá — ver el documento de contexto): Kedro + MongoDB (única base) + LightGBM + SHAP + MLflow local + FastAPI/Docker + Cloud Run + Cloud Logging/Monitoring + Evidently local + GitHub Actions.

Dataset: Online Retail II (Kaggle/UCI).

## M0 — Scaffolding

Estructura completa del proyecto (todas las carpetas de pipelines de Kedro, configuración, Docker, CI) con stubs vacíos. Cero lógica de negocio, cero conexión real a datos. Ver el prompt completo más abajo.

## M1 — Ingesta

Descargar Online Retail II, cargarlo tal cual (sin limpiar) en una colección `raw_transactions` de MongoDB vía el pipeline `ingestion` de Kedro. Objetivo: confirmar que la conexión y la carga funcionan, con un chequeo simple de cantidad de filas cargadas vs. filas del archivo original.

## M2 — Validación y limpieza

Acá vive el nodo explícito de validación que acordamos. Definir reglas concretas (ejemplos, ajustar según lo que se vea en los datos):
- Facturas con prefijo "C" = cancelaciones — decidir si se excluyen del set de entrenamiento o se tratan aparte.
- Cantidades o precios negativos/cero — flaggear, no descartar en silencio.
- Filas sin `CustomerID` (~25% del dataset) — no sirven para churn a nivel cliente, hay que decidir si se descartan o se guardan aparte para otro análisis.
- Fechas fuera de rango o duplicados exactos.

El nodo debe producir dos salidas: una colección `clean_transactions` en Mongo, y un **reporte de validación** (qué regla descartó cuántas filas y por qué) — es el artefacto que después sirve para el Substack.

## M3 — Feature engineering

Calcular por cliente: recencia (días desde última compra), frecuencia (número de pedidos), monetario (ticket promedio/total), categoría favorita, y las que tengan sentido según lo que permita el dataset.

**Dos decisiones técnicas importantes acá:**
1. **Definir el umbral de churn** (ej. "sin compra en los últimos N días") de forma justificada con la distribución real de tiempos entre compras del dataset, no arbitraria.
2. **Punto-en-el-tiempo correcto (evitar leakage):** las features de un cliente para una fecha de corte dada solo pueden usar pedidos *anteriores* a esa fecha. Si se calculan features usando todo el historial (incluyendo compras futuras respecto al punto que se está evaluando), el modelo va a parecer mucho mejor de lo que realmente es. Esto hay que armarlo bien acá, porque es difícil de corregir después.

## M4 — Entrenamiento (LightGBM + MLflow)

Split **temporal**, no aleatorio (por la razón de M3: el churn depende del tiempo). Entrenar LightGBM, trackear parámetros/métricas/modelo en MLflow local. El churn suele venir desbalanceado (pocos clientes churneados vs. muchos activos) — priorizar métricas como PR-AUC o recall a precisión fija por sobre accuracy simple, que en desbalance engaña.

## M5 — Explicabilidad (SHAP)

Sobre el modelo ya entrenado y registrado: valores SHAP por predicción individual (para poder decir "este cliente está en riesgo por X, Y, Z") y un reporte de importancia global de features.

## M6 — Servido (FastAPI + Docker)

Endpoint `POST /predict`: recibe `customer_id`, busca sus features en Mongo, predice, guarda la predicción (con timestamp) en una colección `predictions`, devuelve `churn_probability` + nivel de riesgo + (opcional) la explicación SHAP resumida. Dockerizar y probar local antes de M7.

## M7 — Deployment (Cloud Run + CI)

Deploy de la imagen a Cloud Run. GitHub Actions: lint + test + build de imagen (+ deploy automático si querés ir a fondo). Smoke test simple contra el endpoint ya desplegado.

## M8 — Observabilidad (logging, monitoreo, drift)

Cloud Logging desde FastAPI (logs estructurados, no solo prints). Cloud Monitoring para latencia/requests/errores. Evidently local generando reportes de drift periódicos (comparando distribución de inferencias actuales vs. baseline de entrenamiento), guardados como HTML en Cloud Storage. Este milestone es, en términos de negocio, la prueba tangible de la promesa de fiabilidad.
