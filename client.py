import asyncio
import pygame
import json # for testing
import time
import logging

logging.basicConfig(level=logging.DEBUG)

class Client:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        # connect to server
        self.reader,self.writer = asyncio.open_connection(self.host, self.port)

        self.server_events = []

        pygame.init()
        pygame.display.set_caption("Multiplayer")
        self.HEIGHT = 600
        self.WIDTH = 800
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.fps = 60

        self.running = True

        self.background_color = (0,0,0)

        # for testing
        self.player_pos = (0,0)
        self.player_color = (255,255,255)
        self.player_size = 10

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
            await asyncio.sleep(0)

            self.screen.fill((0,0,0))
            # process input
            messages = []
            for event in pygame.event.get():
                messages.append(self.process_input_event(event))

            # send input to server
            for message in messages:
                self.writer.write(json.dumps(message).encode())
                await self.writer.drain()

            while self.server_events:
                event = self.server_events.pop()
                self.process_server_event(event)

            # draw game state
            pygame.draw.circle(self.screen,self.player_color,self.player_pos,self.player_size)
            pygame.display.update()


    def process_input_event(self,event)->dict:
        # return a dict with the event type and the data
        # wasd send move message
        # mouse move send steer message
        # mouse click send shoot message
        # quit send quit message
        # if event.type == pygame.MOUSEMOTION:
        #     face = event.pos - self.player_pos
        #     {"type":"move","face":face}
        if event.type == pygame.KEYDOWN:
            if event.key in [pygame.K_w,pygame.K_s,pygame.K_a,pygame.K_d]:
                move_x,move_y = 0,0
                if event.key == pygame.K_w:
                    move_y = -1
                elif event.key == pygame.K_s:
                    move_y = 1
                elif event.key == pygame.K_a:
                    move_x = -1
                elif event.key == pygame.K_d:
                    move_x = 1
                message={"type":"move","move":(move_x,move_y)}
        # elif event.type == pygame.MOUSEBUTTONDOWN:
        elif event.type == pygame.QUIT:
            message={"type":"quit"}
            self.running = False
        return message

    def process_server_event(self,event):
        # update game state based on server event and draw it
        if event["type"] == "move":
            self.player_pos = event["pos"]
        elif event["type"] == "quit":
            self.running = False
            

if __name__ == "__main__":
    HOST='localhost'
    PORT=8080
    client = Client(HOST,PORT)
    client.run()