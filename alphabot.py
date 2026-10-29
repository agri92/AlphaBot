# ============================================================
# Chatbot basado en reglas con NLTK - Instituto
# ============================================================

!pip install nltk -q

import re
import random
import unicodedata
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer

# ------------------------------------------------------------
# 1. Configuración inicial de herramientas NLP
# ------------------------------------------------------------

tokenizer = RegexpTokenizer(r'\w+')
stemmer = SnowballStemmer("spanish")

def normalizar_texto(texto):
    texto = texto.lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = texto.encode("ascii", "ignore").decode("utf-8")
    texto = re.sub(r"[^a-zñ0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

def tokenizar_y_stem(texto):
    texto = normalizar_texto(texto)
    tokens = tokenizer.tokenize(texto)
    stems = [stemmer.stem(token) for token in tokens]
    return tokens, stems

# ------------------------------------------------------------
# 2. Definición de las 5 Intenciones del Instituto
# ------------------------------------------------------------

intenciones = {
    "horarios": {
        "ejemplos": [
            "cual es el horario de atencion",
            "a que hora abren",
            "horarios",
            "horario",
            "que dias atienden"
        ],
        "respuestas": [
            "Nuestro horario de atención general es de lunes a viernes de 9:00 AM a 9:00 PM y sábados de 9:00 AM a 2:00 PM."
        ]
    },
    "cursos": {
        "ejemplos": [
            "que cursos tienen disponibles",
            "quiero ver los cursos",
            "oferta academica",
            "cuales son los cursos",
            "tienen diplomados"
        ],
        "respuestas": [
            "Actualmente ofrecemos los siguientes cursos de inglés: Curso regular, Curso de conversación, EXCI, TOEFL. ¿Te interesa alguno en específico?"
        ]
    },
    "agendar_clase": {
        "ejemplos": [
            "quiero agendar mi clase",
            "reservar clase",
            "apartar lugar",
            "agendar espacio",
            "quiero inscribirme a una clase"
        ],
        "respuestas": [
            "Para agendar tu clase, por favor indícame tu matrícula y tu nombre completo para asignarte un espacio disponible."
        ]
    },
    "estado_cuenta": {
        "ejemplos": [
            "como consulto mi estado de cuenta",
            "ver mis pagos",
            "saldo pendiente",
            "cuanto debo",
            "historial de pagos"
        ],
        "respuestas": [
            "Puedes revisar tu saldo actual e historial de pagos ingresando al portal de alumnos en la sección de Mi estado de cuenta."
        ]
    },
    "otro_servicio": {
        "ejemplos": [
            "necesito otra ayuda",
            "hablar con asesor",
            "tengo un problema diferente",
            "ayuda con otro trámite",
            "soporte técnico"
        ],
        "respuestas": [
            "Entiendo. Para brindarte una atención más personalizada con otro servicio, te invito a comunicarte a nuestra línea de atención a clientes."
        ]
    }
}

# ------------------------------------------------------------
# 3. Preparación de vocabulario por intención
# ------------------------------------------------------------

def construir_vocabulario_intenciones(intenciones):
    vocabulario = {}
    for nombre_intencion, datos in intenciones.items():
        stems_intencion = []
        for ejemplo in datos["ejemplos"]:
            _, stems = tokenizar_y_stem(ejemplo)
            stems_intencion.extend(stems)
        vocabulario[nombre_intencion] = set(stems_intencion)
    return vocabulario

vocabulario_intenciones = construir_vocabulario_intenciones(intenciones)

# ------------------------------------------------------------
# 4. Detección de intención
# ------------------------------------------------------------

def detectar_intencion(mensaje):
    _, stems_mensaje = tokenizar_y_stem(mensaje)
    stems_mensaje = set(stems_mensaje)

    puntajes = {}
    for intencion, stems_intencion in vocabulario_intenciones.items():
        coincidencias = stems_mensaje.intersection(stems_intencion)
        puntajes[intencion] = len(coincidencias)

    mejor_intencion = max(puntajes, key=puntajes.get)
    mejor_puntaje = puntajes[mejor_intencion]

    if mejor_puntaje == 0:
        return "otro_servicio", 0  # Redirige por defecto si no hay coincidencia

    return mejor_intencion, mejor_puntaje

# ------------------------------------------------------------
# 5. Generación de respuesta
# ------------------------------------------------------------

def generar_respuesta(mensaje):
    intencion, puntaje = detectar_intencion(mensaje)

    if intencion == "otro_servicio" and puntaje == 0:
        respuesta = "Entiendo. Para brindarte una atención más personalizada con otro servicio, te invito a comunicarte con nuestra línea de soporte directo o acudir a ventanilla escolar."
    else:
        respuesta = random.choice(intenciones[intencion]["respuestas"])

    return {
        "intencion": intencion,
        "puntaje": puntaje,
        "respuesta": respuesta
    }

# ------------------------------------------------------------
# 6. Prueba interactiva del chatbot
# ------------------------------------------------------------

def iniciar_chatbot():
    historial = []

    print("Chatbot: ¡Hola! Bienvenido al servicio de atención a clientes del instituto. ¿En qué puedo ayudarte?")
    print("Chatbot: Escribe 'salir' para terminar.\n")

    while True:
        mensaje_usuario = input("Usuario: ")

        if normalizar_texto(mensaje_usuario) == "salir":
            print("Chatbot: ¡Gracias por comunicarte con el instituto! Que tengas un excelente día.")
            break

        resultado = generar_respuesta(mensaje_usuario)

        historial.append({
            "mensaje_usuario": mensaje_usuario,
            "intencion_detectada": resultado["intencion"],
            "puntaje": resultado["puntaje"],
            "respuesta": resultado["respuesta"]
        })

        print(f"Chatbot: {resultado['respuesta']}\n")

    return pd.DataFrame(historial)

# Ejecutar chatbot
historial_chat = iniciar_chatbot()
historial_chat
