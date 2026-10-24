# Chatbot de Pedidos - El Crustáceo Cascarudo 🍔⚓️

## Descripción del Proyecto
Este repositorio contiene la implementación de un Asistente Inteligente (Chatbot) basado en reglas para resolver un problema de saturación y mala atención en la línea telefónica del restaurante **El Crustáceo Cascarudo**. El sistema utiliza Procesamiento de Lenguaje Natural (NLP) clásico con la librería **NLTK** (tokenización y *stemming*) para clasificar intenciones, extraer entidades (platillos y servicios del menú canónico) y realizar análisis de polaridad de sentimiento básico.

Este proyecto corresponde al Avance 1 (Semana 2) de la Maestría en Inteligencia Artificial.

## Estructura del Repositorio
- `README.md`: Documentación e instrucciones del proyecto.
- `chatbot_crustaceo.ipynb`: Cuaderno de Google Colab con el código fuente en Python.
- `pruebas/casos.csv`: Registro tabular de las 10 conversaciones de prueba (incluyendo los análisis de fallback y falsos positivos).
- `pruebas/capturas/`: Evidencias gráficas (screenshots) de la ejecución del chatbot en el entorno Colab.

## Cómo ejecutar el proyecto
1. Abre [Google Colab](https://colab.research.google.com/).
2. Sube el archivo `chatbot_crustaceo.ipynb` o ábrelo directamente conectando este repositorio de GitHub.
3. Ejecuta la primera celda para instalar y descargar las dependencias necesarias (`nltk`, `pandas`).
4. Corre todas las celdas secuencialmente de arriba hacia abajo.
5. Al final del cuaderno, localiza la celda interactiva, descomenta la llamada a la función `iniciar_chatbot()` y ejecútala para interactuar en vivo con el sistema a través de la consola. Escribe "salir" para terminar la simulación.

## Dependencias Técnicas
- Python 3.10 o superior
- NLTK (Natural Language Toolkit)
- Pandas (para análisis de métricas y exportación de casos)

## Autor
**Mauro Cesar Faz Turrubiate**  
Matrícula: al07319285