import asyncio
import time
import logging

logging.basicConfig(level=logging.DEBUG)

async def func():
    # time.sleep(1)
    await asyncio.sleep(10)

async def main():
    await func()


asyncio.run(main(), debug=True)