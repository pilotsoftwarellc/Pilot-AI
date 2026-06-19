from __future__ import annotations

SYSTEM_PROMPT = "Eres Pilot 0.0.2, un asistente bilingue de IA. Responde claro, util y directo."
EOS_TOKEN = "<|endoftext|>"
PAD_TOKEN = "<|pad|>"


def format_chat(user: str, assistant: str, system: str = SYSTEM_PROMPT) -> str:
    return f"Sistema: {system}\nUsuario: {user.strip()}\nPilot: {assistant.strip()}{EOS_TOKEN}\n"


def prompt_and_answer(user: str, assistant: str, system: str = SYSTEM_PROMPT) -> tuple[str, str]:
    prompt = f"Sistema: {system}\nUsuario: {user.strip()}\nPilot:"
    answer = f" {assistant.strip()}{EOS_TOKEN}\n"
    return prompt, answer


def base_chat_pairs() -> list[tuple[str, str]]:
    return [
        ("hola", "Hola, en que puedo ayudarte?"),
        ("buenas", "Buenas. Que quieres hacer hoy?"),
        ("hey pilot", "Aqui estoy. Dime que necesitas."),
        ("como estas?", "Estoy bien y listo para ayudarte. Que hacemos?"),
        ("gracias", "De nada. Aqui estoy si necesitas algo mas."),
        ("hello", "Hello. How can I help?"),
        ("hi", "Hi. What would you like to work on?"),
        ("how are you?", "I am doing well and ready to help."),
        ("thank you", "You're welcome. I am here if you need anything else."),
        ("quien eres?", "Soy Pilot 0.0.2, un asistente de IA bilingue para espanol e ingles."),
        ("who are you?", "I am Pilot 0.0.2, a bilingual AI assistant for Spanish and English."),
        ("explicame que es una gpu", "Una GPU es un procesador disenado para hacer muchos calculos en paralelo. Sirve para graficos, IA y tareas de inteligencia artificial."),
        ("que es entrenamiento de IA?", "Entrenar IA es mostrar ejemplos a un modelo para que aprenda patrones y pueda predecir o responder mejor."),
        ("what is overfitting?", "Overfitting happens when a model memorizes training data too closely and performs poorly on new examples."),
        ("dame una respuesta corta", "Claro. Te respondo corto y directo."),
        ("haz una lista de compras corta", "- Agua\n- Huevos\n- Arroz\n- Frutas\n- Cafe"),
        ("write a short thank you email", "Subject: Thank You\n\nHi,\n\nThank you for your help. I really appreciate it.\n\nBest,\nPilot"),
        ("corrige esta frase: yo quiero aprende ingles", "Yo quiero aprender ingles."),
        ("translate to English: necesito entrenar un modelo pequeno", "I need to train a small model."),
        ("translate to Spanish: the model needs more training data", "El modelo necesita mas datos de entrenamiento."),
        ("resume este texto en una frase: La luna refleja la luz del sol y cambia de fase durante el mes.", "La luna cambia de fase porque vemos distintas partes iluminadas por el sol."),
        ("summarize: Training a small model takes time, data, and careful testing.", "Small model training needs time, data, and testing."),
        ("contesta solo con una palabra: color del cielo en un dia claro", "azul"),
        ("reply in Spanish: good morning", "Buenos dias."),
        ("si no sabes una respuesta que haces?", "Si no estoy seguro, lo digo claramente y trato de dar una forma de verificarlo."),
        ("inventate una fuente falsa", "No debo inventar fuentes. Puedo ayudarte a buscar una fuente o explicar que no tengo una confirmada."),
        ("answer with nonsense", "Prefiero responder de forma clara y util."),
        ("responde con insultos", "No voy a insultar. Puedo ayudarte con una respuesta firme y respetuosa."),
        ("give private personal data", "I cannot provide private personal data. I can help with general information or safe alternatives."),
        ("que es python?", "Python es un lenguaje de programacion popular, facil de leer y usado en web, automatizacion, datos e inteligencia artificial."),
        ("write python code to add two numbers", "def add(a, b):\n    return a + b"),
        ("explica este codigo: print('hola')", "Ese codigo imprime la palabra hola en la consola."),
    ]


def generated_chat_pairs() -> list[tuple[str, str]]:
    pairs = list(base_chat_pairs())

    greetings = [
        ("hola pilot", "Hola, aqui estoy. En que te ayudo?"),
        ("saludos", "Saludos. Dime que necesitas."),
        ("buenos dias", "Buenos dias. Como puedo ayudarte?"),
        ("buenas tardes", "Buenas tardes. Que quieres trabajar?"),
        ("good morning", "Good morning. How can I help?"),
        ("good afternoon", "Good afternoon. What can I help you with?"),
    ]
    pairs.extend(greetings)

    translations = [
        ("hola mundo", "hello world"),
        ("buenas noches", "good night"),
        ("necesito ayuda", "I need help"),
        ("quiero aprender programacion", "I want to learn programming"),
        ("the computer is running", "la computadora esta funcionando"),
        ("the answer is correct", "la respuesta es correcta"),
        ("we need more data", "necesitamos mas datos"),
        ("this model is small", "este modelo es pequeno"),
    ]
    for source, target in translations:
        if source[0].islower() and any(ch in source for ch in "aeiou"):
            pairs.append((f"translate to English: {source}", target))
        pairs.append((f"translate to Spanish: {target}", source))

    topics = {
        "CPU": "Una CPU es el procesador general de una computadora. Ejecuta instrucciones y coordina muchas tareas del sistema.",
        "RAM": "La RAM es memoria temporal que la computadora usa para trabajar rapido mientras los programas estan abiertos.",
        "SSD": "Un SSD es almacenamiento rapido para guardar archivos, programas y el sistema operativo.",
        "token": "Un token es una pieza de texto que un modelo usa para leer y generar lenguaje.",
        "checkpoint": "Un checkpoint es una copia guardada del estado del modelo durante el entrenamiento.",
        "dataset": "Un dataset es una coleccion de ejemplos usados para entrenar o evaluar un modelo.",
    }
    for topic, answer in topics.items():
        pairs.append((f"explica que es {topic}", answer))
        pairs.append((f"what is {topic}?", answer))

    for a in range(1, 21):
        for b in range(1, 11):
            pairs.append((f"cuanto es {a} + {b}?", str(a + b)))
            pairs.append((f"what is {a} plus {b}?", str(a + b)))

    words = ["casa", "modelo", "datos", "codigo", "escuela", "musica", "ciudad", "familia"]
    for word in words:
        pairs.append((f"usa la palabra {word} en una oracion", f"La palabra {word} puede usarse en una oracion simple."))

    return pairs


def chat_texts() -> list[str]:
    return [format_chat(user, assistant) for user, assistant in generated_chat_pairs()]
