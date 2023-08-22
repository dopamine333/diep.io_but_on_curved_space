import asyncio
import pygame
import json # for testing
import time
import logging

logging.basicConfig(level=logging.DEBUG)

class Server:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        # connect to server
        asyncio.start_server(self.handle_client, self.host, self.port)

        self.clock = pygame.time.Clock()
        self.fps = 60

        self.running = True

        self.background_color = (0,0,0)

        self.rooms = []

        # for testing
        self.players = []
        self.player_pos = (0,0)
        self.player_color = (255,255,255)
        self.player_size = 10

    def handle_client(self, reader, writer):
        pass

    def run(self):
        asyncio.create_task(self.listen())
        asyncio.run(self.gameloop())
        # try:
        #     asyncio.create_task(self.listen())
        #     asyncio.run(self.gameloop())
        # finally:
        #     self.writer.close()
        #     print("Connection closed")

    async def listen(self):
        while True:
            data = await self.reader.read(1024)
            if not data:
                break
            print(data.decode())
            # call some function to handle the data
            # maybe we should use a different format than json, pickle is faster ?
            event = json.loads(data.decode())
            self.server_events.append(event)
        self.writer.close()
    
    async def gameloop(self):
        # current_time = time.time()
        while self.running:
            # last_time, current_time = current_time, time.time()
            # await asyncio.sleep(1 / self.fps - (current_time - last_time))   

            self.clock.tick(self.fps)

            self.screen.fill(self.background_color)

            for event in self.server_events:
                if event["type"] == "player":
                    self.player_pos = event["pos"]
                    self.player_color = event["color"]
                    self.player_size = event["size"]

            pygame.draw.circle(self.screen, self.player_color, self.player_pos, self.player_size)

            pygame.display.update()

            self.server_events = []

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        pass