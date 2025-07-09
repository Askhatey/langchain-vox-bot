import os
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

from Agent import Agent

# Загрузка переменных окружения из .env файла
load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Инициализация агента
try:
    agent = Agent()
    logger.info("Агент успешно инициализирован.")
except Exception as e:
    logger.error(f"Ошибка при инициализации агента: {e}")
    agent = None

@app.get("/")
async def get():
    return HTMLResponse("<h1>Агент активен. Подключитесь через WebSocket по адресу /ws</h1>")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Клиент подключился по WebSocket.")
    if not agent:
        await websocket.send_text("Ошибка: Агент не был инициализирован. Проверьте логи сервера.")
        await websocket.close()
        logger.error("Попытка подключения при неинициализированном агенте.")
        return
        
    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"Получен вопрос: {data}")
            
            # Используем генератор для потоковой передачи ответа
            response_generator = agent.get_response_generator(data)
            
            full_response = ""
            for chunk in response_generator:
                # Отправляем каждый фрагмент по WebSocket
                await websocket.send_text(chunk)
                full_response += chunk
            
            logger.info(f"Полный ответ отправлен: {full_response}")

    except WebSocketDisconnect:
        logger.info("Клиент отключился.")
    except Exception as e:
        logger.error(f"Произошла ошибка в WebSocket: {e}")
        # Попытка уведомить клиента об ошибке, если соединение еще открыто
        try:
            await websocket.send_text(f"Произошла внутренняя ошибка сервера: {e}")
        except Exception as close_e:
            logger.error(f"Не удалось отправить сообщение об ошибке клиенту: {close_e}")
