import asyncio
import websockets
import sys

async def test_agent():
    """
    Тестовый клиент для подключения к WebSocket.
    """
    uri = "ws://127.0.0.1:8000/ws"
    try:
        async with websockets.connect(uri) as websocket:
            question = "Расскажи про тариф Супер Smart"
            print(f">>> Клиент: {question}")
            
            await websocket.send(question)
            
            # Получаем потоковый ответ от сервера
            print("<<< Сервер:")
            full_response = ""
            try:
                while True:
                    response_part = await asyncio.wait_for(websocket.recv(), timeout=15.0)
                    print(response_part, end="", flush=True)
                    full_response += response_part
            except asyncio.TimeoutError:
                print("\n\n(Тайм-аут: Сервер не прислал новых частей ответа за 15 секунд)")
            except websockets.exceptions.ConnectionClosed:
                print("\n\n(Соединение успешно закрыто сервером)")


    except websockets.exceptions.ConnectionClosed as e:
        print(f"Соединение закрыто: {e.code} {e.reason}")
    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {e}", file=sys.stderr)

if __name__ == "__main__":
    asyncio.run(test_agent()) 