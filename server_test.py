import time
import pygame
import numpy as np
import sympy as sp
from scipy.integrate import odeint
import pickle
import asyncio

import logging
logging.basicConfig(level=logging.DEBUG)

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0,0,255)
GRAY = (128, 128, 128)

class SqueezableCircle:
    def __init__(self,edge_num:int) -> None:
        theta = np.linspace(0, 2*np.pi, edge_num).reshape(-1,1)
        self.cos = np.cos(theta)
        self.sin= np.sin(theta)

    def get_vertex(self,center:np.ndarray,right:np.ndarray,up:np.ndarray,radius:float):
        return center + (self.cos*right+self.sin*up)*radius
    
    def check_collision(self,map:'Map',
                        center:np.ndarray,right:np.ndarray,up:np.ndarray,radius:float,
                        other_center:np.ndarray,other_right:np.ndarray,other_up:np.ndarray,other_radius:float):
        vertex = self.get_vertex(center,right,up,radius)
        vector = vertex - other_center
        vector = vector.T
        in_other = map.dot(other_center,vector,vector) < other_radius**2
        return np.any(in_other)
    
SQUEEZABLE_CIRCLE = SqueezableCircle(20)
COLLISION_SQUEEZABLE_CIRCLE = SqueezableCircle(8)

class Bullet:
    def __init__(self,
                 pos: np.ndarray,
                 vel: np.ndarray,
                 lifetime: float,
                 map: 'Map',
                 color,
                 radius,
                 owner:'Player') -> None:
        self.pos = pos
        # vel is components of velocity in global coordinates
        self.vel = vel
        self.map = map
        
        self.up= self.map.normalize(self.pos, self.vel)
        self.right = self.map.rotate90(self.pos, self.up)

        self.speed_inv = 1/np.sqrt(self.map.dot(self.pos, self.vel, self.vel))

        self.color = color
        self.current_color = color
        self.radius = radius

        self.lifetime = lifetime
        self.alive = True

        self.owner = owner

    def update(self, dt):
        if not self.alive:
            return
        new_vel = self.map.parallel_transport(self.pos, self.vel, self.vel, dt)
        self.pos += self.vel*dt
        self.vel = new_vel
        self.up = self.vel*self.speed_inv
        self.right = self.map.rotate90(self.pos, self.up)

        self.lifetime -= dt
        if self.lifetime <= 0:
            self.alive = False

        self.current_color = self.color
        

        # print pos round to 2 decimal places
        # print(f"pos: {self.pos.round(2)}")
        # print(f"vel len: {self.map.dot(self.pos, self.vel,self.vel)}")

    def check_collision(self,player:'Player'):
        if self.owner == player:
            return
        # check collision with player
        if COLLISION_SQUEEZABLE_CIRCLE.check_collision(self.map,
                                                self.pos, self.right, self.up, self.radius,
                                                player.pos, player.right, player.up, player.radius):
            self.current_color = np.array(WHITE)-self.color


    def draw(self, camera: 'Camera'):
        # vel_factor = 0.5
        # up = self.map.normalize(self.pos, self.vel)

        # vertex = SQUEEZABLE_CIRCLE.get_vertex(self.pos,right,up,self.radius)
        # camera.draw_polygon(self.color, vertex)
        camera.draw_squeezable_circle(
            self.current_color, self.pos, self.right,self.up,self.radius)


        # draw line to show local coordinate
        coord_show_factor = self.radius
        camera.draw_line((0, 0, 255), self.pos,
                         self.pos+self.vel*coord_show_factor)

class Player:
    def __init__(self,
                 pos: np.ndarray,
                 vel: np.ndarray,
                 speed: float,
                 bullet_speed: float,
                 bullet_radius: float,
                 bullet_lifetime: float,
                 map: 'Map',
                 color,
                 radius,
                 controller:'NetworkControllerReader',
                 game:'Game') -> None:
        self.pos = pos
        # vel is components of velocity in local coordinates
        self.vel = vel
        self.speed = speed
        self.map = map

        self.bullet_speed = bullet_speed
        self.bullet_radius = bullet_radius
        self.bullet_lifetime = bullet_lifetime

        # local coordinate
        self.right, self.up = self.random_orthogonal_coordinates()

        self.color = color
        self.radius = radius

        self.controller = controller
        self.game = game
        self.camera = None

    def set_camera(self,camera:'Camera'):
        self.camera = camera

    def restart(self,pos:np.ndarray):
        self.pos = pos
        self.vel = np.array([0, 0])
        self.right, self.up = self.random_orthogonal_coordinates()

    def random_orthogonal_coordinates(self):
        # right = np.random.rand(2)
        right = np.array([1, 0])
        right = self.map.normalize(self.pos, right)
        return right, self.map.rotate90(self.pos, right)

    def shoot(self, mouse_pos):
        # mouse_pos is the position of mouse in global coordinates
        # calculate the velocity of bullet in global coordinates
        direction = mouse_pos - self.pos
        direction = self.map.normalize(self.pos, direction)
        bullet_vel = self.bullet_speed * direction
        bullet = Bullet(
            pos=self.pos.copy(),
            vel=bullet_vel,
            map=self.map,
            color=self.color,
            radius=self.bullet_radius,
            lifetime=self.bullet_lifetime,
            owner=self)
        self.game.add_bullet(bullet)

    def update(self, dt):
        shoot = self.controller.get_jump() or self.controller.get_fire()
        if shoot:
            # print("shoot")
            screen_mouse_pos = self.controller.get_mouse_pos()
            world_mouse_pos = self.camera.screen_to_world(screen_mouse_pos)
            self.shoot(world_mouse_pos)

        x = self.controller.get_horizontal()
        y = self.controller.get_vertical()
        vel = self.speed * (x*self.right + y*self.up)

        self.up = self.map.parallel_transport(
            self.pos, vel, self.up, dt)
        self.right = self.map.parallel_transport(
            self.pos, vel, self.right, dt)

        self.pos += vel*dt
        # print pos round to 2 decimal places
        # print(f"pos: {self.pos.round(2)}")
        # print(f"up len: {self.map.dot(self.pos, self.up, self.up)}")
        # print(f"right len: {self.map.dot(self.pos, self.right, self.right)}")
        # print(f"up dot right: {self.map.dot(self.pos, self.up, self.right)}")

    def draw(self, camera: 'Camera'):
        # vertex = SQUEEZABLE_CIRCLE.get_vertex(self.pos,self.right,self.up,self.radius)
        # camera.draw_polygon(self.color, vertex)
        camera.draw_squeezable_circle(self.color, self.pos, self.right,self.up,self.radius)

        # draw line to show local coordinate
        coord_show_factor = self.radius
        camera.draw_line( RED, self.pos, self.pos+self.right*coord_show_factor)
        camera.draw_line( GREEN, self.pos, self.pos+self.up*coord_show_factor)

        # draw shoot preview
        # screen_mouse_pos = pygame.mouse.get_pos()
        # world_mouse_pos = camera.screen_to_world(screen_mouse_pos)
        # direction = world_mouse_pos - self.pos
        # direction = self.map.normalize(self.pos, direction)
        # trajectory=self.map.geodesic(self.pos, direction, self.bullet_speed, 20)[:,:2]
        # camera.draw_lines(BLUE, False, trajectory)

class Map:
    def __init__(self, x: sp.Symbol, y: sp.Symbol, metric: list[list[sp.Symbol]]):
        self.x, self.y = x, y
        self.coord = [self.x, self.y]

        self.metric = metric
        self.metric_inv: list[list[sp.Symbol]
                              ] = sp.Matrix(metric).inv().tolist()

        self.chris: list[list[list[sp.Symbol]]] = self.calculate_chris()

    def calculate_chris(self):
        '''
        $$\Gamma _{{kl}}^{i}={\frac  {1}{2}}g^{{im}}\left({\frac  {\partial g_{{mk}}}{\partial x^{l}}}+{\frac  {\partial g_{{ml}}}{\partial x^{k}}}-{\frac  {\partial g_{{kl}}}{\partial x^{m}}}\right)$$
        '''
        chris = [[[0 for l in range(len(self.coord))] for k in range(
            len(self.coord))] for i in range(len(self.coord))]
        for i in range(len(self.coord)):
            for k in range(len(self.coord)):
                for l in range(len(self.coord)):

                    chris[i][k][l] = sum(

                        sp.Rational(1, 2)*self.metric_inv[i][m]*(
                            self.metric[m][k].diff(self.coord[l])
                            + self.metric[m][l].diff(self.coord[k])
                            - self.metric[k][l].diff(self.coord[m])
                        )

                        for m in range(len(self.coord)))

                    chris[i][k][l] = sp.simplify(chris[i][k][l])
        return chris

    def parallel_transport(self, pos, vel, vec, dt):
        r'''
        $$\nabla_{\vec{w}}\vec{v}=0$$
        $$\left(\frac{d v^k}{d \lambda}+v^iw^j\Gamma^k_{ij}\right)\vec{e}_k=\vec{0}$$
        $$\dot{v}^k=-v^iw^j\Gamma^k_{ij}$$
        '''
        # v -> vector
        # w -> velocity
        v1, v2, w1, w2 = sp.symbols('v1 v2 w1 w2')
        v = [v1, v2]
        w = [w1, w2]
        v_d: list[sp.Symbol] = [0 for k in range(len(self.coord))]
        for k in range(len(self.coord)):
            v_d[k] = sum(sum(

                -v[i]*w[j]*self.chris[k][i][j]

                for i in range(len(self.coord))) for j in range(len(self.coord)))

            v_d[k] = sp.simplify(v_d[k])

        delta_t = sp.symbols('delta_t')
        v_next = [0 for k in range(len(self.coord))]
        for k in range(len(self.coord)):
            v_next[k] = v[k]+v_d[k]*delta_t
            v_next[k] = sp.simplify(v_next[k])

        self.parallel_transport = sp.lambdify([(self.x, self.y), (w1, w2), (v1, v2) ,delta_t], sp.Array(v_next), modules='numpy')
        return self.parallel_transport(pos, vel, vec, dt)

    def dot(self, pos, vec1, vec2):
        v1, v2, w1, w2 = sp.symbols('v1 v2 w1 w2')
        v = [v1, v2]
        w = [w1, w2]
        dot = sum(sum(

            v[i]*w[j]*self.metric[i][j]

            for i in range(len(self.coord))) for j in range(len(self.coord)))
        dot = sp.simplify(dot)
        self.dot = sp.lambdify(
            [(self.x, self.y), (v1, v2), (w1, w2)], dot, modules='numpy')
        return self.dot(pos, vec1, vec2)

    def rotate90(self,pos, vec):
        # algebraic_multiplicity is 1 or 2
        # eigenvects returns a list of tuples of the form (eigenvalue, algebraic_multiplicity, [eigenvectors])
        vects = sp.Matrix(self.metric).eigenvects()
        algebraic_multiplicity = vects[0][1]
        if algebraic_multiplicity == 1:
            v1 = vects[0][2][0]
            v2 = vects[1][2][0]
        elif algebraic_multiplicity == 2:
            v1 = vects[0][2][0]
            v2 = vects[0][2][1]
        else:
            raise ValueError('algebraic_multiplicity is not 1 or 2')
        u1 = v1/sp.sqrt((v1.T * sp.Matrix(self.metric) * v1)[0])
        u2 = v2/sp.sqrt((v2.T * sp.Matrix(self.metric) * v2)[0])
        M = sp.Matrix.hstack(u1, u2)
        Rot = sp.Matrix(
            [[0, -1], 
             [1, 0]])
        R_tilde = M*Rot*M.inv()
        R_tilde.simplify()
        w1, w2 = sp.symbols('w1 w2')
        w = sp.Matrix([w1, w2])
        R_w = R_tilde@w
        R_w = sp.simplify(R_w)
        self.rotate90 = sp.lambdify([(self.x, self.y), (w1, w2)], sp.Array([R_w[0], R_w[1]]), modules='numpy')
        return self.rotate90(pos, vec)

    def normalize(self, pos, vec):
        return vec/np.sqrt(self.dot(pos, vec, vec))

    def geodesic(self, pos, vel, time, step):
        # Geodesic equations
        t=sp.symbols('t')
        gamma1,gamma2=sp.symbols(r'\gamma^1 \gamma^2',cls=sp.Function)
        gamma1,gamma2=gamma1(t),gamma2(t)
        gamma=[gamma1,gamma2]
        geodesic_eq=[0 for lam in range(len(gamma))]
        geodesic_chris=[[[

            self.chris[i][j][k].subs({self.coord[l]:gamma[l] for l in range(len(self.coord))})
                        
        for k in range(len(self.coord))] for j in range(len(self.coord))] for i in range(len(self.coord))]

        for lam in range(len(gamma)):
            
            geodesic_eq[lam]=gamma[lam].diff(t,t)+sum(sum(

                geodesic_chris[lam][mu][nu]*gamma[mu].diff(t)*gamma[nu].diff(t)

            for mu in range(len(gamma))) for nu in range(len(gamma)))

        sol=sp.solve(geodesic_eq,[gamma1.diff(t,t),gamma2.diff(t,t)])
        deriv1=gamma1.diff(t)
        deriv2=gamma2.diff(t)
        deriv3=sol[gamma1.diff(t,t)]
        deriv4=sol[gamma2.diff(t,t)]
        deriv1,deriv2,deriv3,deriv4
        S=(gamma1,gamma2,gamma1.diff(t),gamma2.diff(t))
        deriv1_num=sp.lambdify(S,deriv1)
        deriv2_num=sp.lambdify(S,deriv2)
        deriv3_num=sp.lambdify(S,deriv3)
        deriv4_num=sp.lambdify(S,deriv4)
        def dSdt(S,t):
            return (deriv1_num(*S),deriv2_num(*S),deriv3_num(*S),deriv4_num(*S))
        
        def geodesic(pos,vel,time,step):
            t_run=np.linspace(0,time,step)
            # print(pos,vel)
            return odeint(dSdt,(*pos,*vel),t_run)
        self.geodesic=geodesic
        return self.geodesic(pos,  vel, time, step)

class MapDrawer:
    def __init__(self, map:'Map', step ,space ,length,color):
        self.map = map
        self.step = step
        self.space = space
        self.length = length
        self.color = color

        self.curves = []

    def generate(self,center,right):
        # calculate x axis and y axis lines (geodesics)
        # on x axis every space, draw vertical lines
        # on y axis every space, draw horizontal lines
        right = self.map.normalize(center,right)
        up = self.map.rotate90(center,right)
        # geodesic.shape = (step,4)
        # geodesic = [(x1,y1,x1',y1'),(x2,y2,x2',y2'),...]
        space_in_index = round(self.space/self.length*self.step)
        if space_in_index == 0:
            space_in_index = 1
        half_length = self.length/2
        half_step = round(self.step/2)+1
        x_axis = np.concatenate(
            (self.map.geodesic(center,-right,half_length,half_step)[:0:-1],
             self.map.geodesic(center,right,half_length,half_step)),axis=0)
        # self.curves.append(x_axis[:, :2]) 
        x_pos = x_axis[::space_in_index, :2]
        x_right = x_axis[::space_in_index, 2:]
        # print(x_axis.shape,x_pos.shape,x_right.shape)
        # print(x_axis,x_pos,x_right)
        x_up = self.map.rotate90(x_pos.T,x_right.T).T  
        # print(x_up.shape)
        # print(x_up)
        y_axis = np.concatenate(
            (self.map.geodesic(center,-up,half_length,half_step)[:0:-1],
             self.map.geodesic(center,up,half_length,half_step)),axis=0)
        # self.curves.append(y_axis[:, :2])
        y_pos = y_axis[::space_in_index, :2]
        y_up = y_axis[::space_in_index, 2:]
        y_right = -self.map.rotate90(y_pos.T,y_up.T).T

        for pos,up in zip(x_pos,x_up):
            vertical = np.concatenate(
                (self.map.geodesic(pos,-up,half_length,half_step)[:0:-1],
                 self.map.geodesic(pos,up,half_length,half_step)),axis=0)
            self.curves.append(vertical[:, :2])
        for pos,right in zip(y_pos,y_right):
            horizontal = np.concatenate(
                (self.map.geodesic(pos,-right,half_length,half_step)[:0:-1],
                 self.map.geodesic(pos,right,half_length,half_step)),axis=0)
            self.curves.append(horizontal[:, :2])

    def draw(self,camera:'Camera'):
        for curve in self.curves:
            camera.draw_lines(self.color,False,curve)
        
class Camera:
    def __init__(self, pos, width, height, zoom,
                  screen :'NetworkSurfaceWriter',
                  player:'Player',
                  controller:'NetworkControllerReader'):
        self.pos = pos
        self.zoom = zoom
        self.width, self.height = width, height
        self.center = np.array([width/2, height/2])
        # self.screen = pygame.display.set_mode((self.width, self.height))
        self.filp =  np.diag([1,-1])
        
        self.screen = screen
        self.player = player
        self.controller = controller


    def update(self):
        if self.controller.get_key_pressed(pygame.K_q):
            self.zoom_in_or_out(0.99)
        if self.controller.get_key_pressed(pygame.K_e):
            self.zoom_in_or_out(1.01)
        self.pos = self.player.pos
        # player_to_world=np.vstack((player.up,player.right)).T
        # self.world_to_screen_matrix = np.linalg.inv(player_to_world)@np.diag([self.zoom,self.zoom])

        up = self.player.up / np.linalg.norm(self.player.up)
        # orthogonal_player_to_world=np.vstack((right,up)).T
        orthogonal_player_to_world=np.array([
            [up[1],up[0]],
            [-up[0],up[1]]]
        )
        # self.world_to_screen_matrix = np.linalg.inv(orthogonal_player_to_world)@np.diag([self.zoom,self.zoom])
        self.screen_to_world_matrix = orthogonal_player_to_world@self.filp/self.zoom
        self.world_to_screen_matrix = np.linalg.inv(self.screen_to_world_matrix)

        self.world_to_screen_matrix = self.world_to_screen_matrix.T
        self.world_to_screen_offset = -self.pos@self.world_to_screen_matrix+self.center

        self.screen_to_world_matrix = self.screen_to_world_matrix.T
        self.screen_to_world_offset = -self.center@self.screen_to_world_matrix+self.pos

        

    def world_to_screen(self, pos):
        # print("self.pos",self.pos)
        # print("pos",pos)
        # return (pos-self.pos)@self.world_to_screen_matrix.T+self.center
        return pos@self.world_to_screen_matrix+self.world_to_screen_offset
    
    def screen_to_world(self,pos):
        # return (pos-self.center)@self.screen_to_world_matrix.T+self.pos
        return pos@self.screen_to_world_matrix+self.screen_to_world_offset
    
    def zoom_in_or_out(self,factor):
        self.zoom *= factor

    def draw_polygon(self, color, points):
        points = self.world_to_screen(points)
        self.screen.draw_polygon(color, points)
        # pygame.draw.polygon(self.screen, color, points)

    def draw_line(self, color, start, end):
        start = self.world_to_screen(start)
        end = self.world_to_screen(end)
        self.screen.draw_line(color, start, end)
        # pygame.draw.line(self.screen, color, start, end)

    def draw_circle(self, color, pos, radius):
        pos = self.world_to_screen(pos)
        # check if circle is on screen
        radius = radius*self.zoom
        if radius < 0.5:
            return
        right_edge = pos[0] + radius
        left_edge = pos[0] - radius
        top_edge = pos[1] + radius
        bottom_edge = pos[1] - radius
        if right_edge < 0 or left_edge > self.width or top_edge < 0 or bottom_edge > self.height:
            return
        self.screen.draw_circle(color, pos, radius)
        # pygame.draw.circle(self.screen, color, pos, radius)

    def draw_circles(self, color, pos_list, radius_list):
        radius_list = radius_list * self.zoom
        threshold = 0.5
        big_enough = radius_list >= threshold # only draw circles that are big enough to see
        pos_list = pos_list[big_enough]
        radius_list = radius_list[big_enough]

        pos_list = self.world_to_screen(pos_list)
        right_edges = pos_list[:,0] + radius_list
        left_edges = pos_list[:,0] - radius_list
        top_edges = pos_list[:,1] + radius_list
        bottom_edges = pos_list[:,1] - radius_list
        on_screen = np.logical_and.reduce((right_edges > 0, left_edges < self.width, top_edges > 0, bottom_edges < self.height))

        pos_list = pos_list[on_screen]
        radius_list = radius_list[on_screen]

        for pos,radius in zip(pos_list,radius_list):
            self.screen.draw_circle(color, pos, radius)
            # pygame.draw.circle(self.screen, color, pos, radius)

    def draw_lines(self, color, closed, points):
        points = self.world_to_screen(points)
        self.screen.draw_lines(color, closed, points)
        # pygame.draw.lines(self.screen, color, closed, points)

    def draw_squeezable_circle(self,color,pos,right,up,radius):
        pos = self.world_to_screen(pos)
        right = right@self.world_to_screen_matrix
        up = up@self.world_to_screen_matrix
        # area = max(np.linalg.norm(right),np.linalg.norm(up))
        # print("area_square",area)
        # if area < 10:
        #     return
        vertex = SQUEEZABLE_CIRCLE.get_vertex(pos,right,up,radius)
        self.screen.draw_polygon(color, vertex)
        # pygame.draw.polygon(self.screen, color, vertex)

    def clear(self):
        self.screen.fill((50, 50, 60))

    def draw(self):
        self.screen.draw()
        # pygame.display.update()

class FPS:
    def __init__(self,clock):
        self.clock = clock
        self.font = pygame.font.SysFont("Verdana", 20)
        self.text = None #self.font.render(str(self.clock.get_fps()), True, BLACK)

    def draw(self, screen):
        self.text = self.font.render("FPS: "+str(round(self.clock.get_fps(),2)), True, WHITE)
        screen.blit(self.text, (200, 150))

    def show_bullets(self, screen, bullets):
        bullet_num_text="bullet number: "+str(len(bullets))
        self.text = self.font.render(bullet_num_text, True, WHITE)
        screen.blit(self.text, (400, 150))

class MouseCircle:
    def __init__(self, camera:'Camera',map:'Map',player:'Player', pos, radius, original_color, collision_color    ):
        self.pos = pos
        self.radius = radius
        self.color = original_color
        self.original_color = original_color
        self.collision_color = collision_color
        self.up = np.array([0,1],dtype=np.float64)
        self.right = np.array([1,0],dtype=np.float64)
        
        self.camera = camera
        self.map = map
        self.player = player

    def update(self):
        self.pos = self.camera.screen_to_world(pygame.mouse.get_pos())
        self.right = self.map.normalize(self.pos,np.array([0,1],dtype=np.float64))
        self.up = self.map.rotate90(self.pos,self.right)

        # check collision with player
        if COLLISION_SQUEEZABLE_CIRCLE.check_collision(self.map,
            self.pos,self.right,self.up,self.radius,
            self.player.pos,self.player.right,self.player.up,self.player.radius):
            self.color = self.collision_color
        else:
            self.color = self.original_color

    def draw(self, camera: 'Camera'):
        camera.draw_squeezable_circle(self.color, self.pos, self.right, self.up, self.radius)

class Server:
    def __init__(self,host,port,game:'Game') -> None:
        self.host = host
        self.port = port
        self.game = game

        self.header_length = 12

        # [(writer,data),...]
        self.send_buffer = []

        # 10 mb
        # self.buffer_limit = 10*1024*1024

        self.send_count=0
        self.recv_count=0

        self.debug = False

        # use to send close event
        self.all_writer: list['asyncio.StreamWriter'] = []
        self.server_task:'asyncio.Server' = None

    async def start_server(self):
        self.server_task = await asyncio.start_server(
            self.handle_client,
            self.host,
            self.port,
            # limit=self.buffer_limit
            )

    async def handle_client(self,reader:'asyncio.StreamReader'
                            , writer:'asyncio.StreamWriter'):
        print(f"new client connented, {writer.get_extra_info('peername')}")
        self.all_writer.append(writer)
        controller = NetworkControllerReader(self, reader)
        screen = NetworkSurfaceWriter(self, writer)
        player = self.game.add_player(controller)
        camera = self.game.add_camera(player, screen, controller)
        player.set_camera(camera)
        self.game.add_controller(controller)
        await controller.listen()
        print(f"client disconnected, {writer.get_extra_info('peername')}")
        # after listen, this client is disconnected
        # clear this client's resources
        self.game.remove_player(player)
        self.game.remove_camera(camera)
        self.game.remove_controller(controller)
        # writer.close()
        # await writer.wait_closed()
        # writer.write(b'see ya')

        self.all_writer.remove(writer)

    def send(self,event,writer:'asyncio.StreamWriter'):
        event_dumps = pickle.dumps(event)
        event_length = len(event_dumps)
        header = str(event_length).encode('utf-8').ljust(self.header_length)
        data = header+event_dumps
        if self.debug:
            self.send_count += 1
            print()
            print(f"> count\t{self.send_count}")
            print(f"> type\t{event['type']}")
            print(f"> length\t{event_length}")
            # print(f"> header\t{header}")
        # writer.write(data)
        self.send_buffer.append((writer,data))

    async def drain(self):
        while self.send_buffer:
            writer, data = self.send_buffer.pop(0)
            writer.write(data)
            # await writer.drain()


    async def read_until(self,length,reader:'asyncio.StreamReader'):
        data = await reader.read(length)
        if not data:
            return None
        if data == b'bye':
            print("bye")
            return None
        while len(data) != length:
            remaining = length - len(data)
            print(f"read remaining:  {length} - {len(data)} = {remaining}")
            data += await reader.read(remaining)
        return data

    async def recv(self,reader:'asyncio.StreamReader'):
        header = await self.read_until(self.header_length,reader)
        if not header:
            return None
        event_length = int(header.decode('utf-8').strip())
        event_dumps = await self.read_until(event_length,reader)
        event = pickle.loads(event_dumps)
        if self.debug:
            self.recv_count += 1
            print()
            print(f"< count\t{self.recv_count}")
            print(f"< type\t{event['type']}")
            print(f"< length\t{event_length}")
            # print(f"< header\t{header}")
        return event
    
    async def close(self):
        self.server_task.close()
        for writer in self.all_writer:
            writer.close()
            await writer.wait_closed()

class NetworkSurfaceWriter:
    # mimic pygame.Surface
    def __init__(self, 
                 server:'Server', 
                 writer:'asyncio.StreamWriter') -> None:
        self.writer = writer
        self.server = server
    
        # pygame.draw.line(self.screen, color, start, end)
        # pygame.draw.lines(self.screen, color, closed, points)
        # pygame.draw.circle(self.screen, color, pos, radius)
        # pygame.draw.polygon(self.screen, color, vertex)
        # pygame.draw.fill(self.screen, color)
        # pygame.display.update()

    def draw_line(self, color, start_pos, end_pos):
        event={
            'type':'draw_line',
            'params':{
                'color':color,
                'start_pos':start_pos,
                'end_pos':end_pos
            }
        }
        self.server.send(event,self.writer)
    
    def draw_lines(self, color, closed, points):
        event={
            'type':'draw_lines',
            'params':{
                'color':color,
                'closed':closed,
                'points':points
            }
        }
        self.server.send(event,self.writer)

    def draw_circle(self, color, center, radius):
        event={
            'type':'draw_circle',
            'params':{
                'color':color,
                'center':center,
                'radius':radius
            }
        }
        self.server.send(event,self.writer)

    def draw_polygon(self, color, points):
        event={
            'type':'draw_polygon',
            'params':{
                'color':color,
                'points':points
            }
        }
        self.server.send(event,self.writer)

    def fill(self, color):
        event={
            'type':'fill',
            'params':{
                'color':color
            }
        }
        self.server.send(event,self.writer)

    def draw(self):
        event={
            'type':'draw',
            'params':{}
        }
        self.server.send(event,self.writer)

class NetworkControllerReader:
    def __init__(self,
                 server:'Server',
                 reader:'asyncio.StreamReader') -> None:
        self.server = server
        self.reader = reader

        self.inputs = {
            'trigger':[
                # 'jump', 'fire'
            ],
            'mouse_pos':None,
            # ad or left right
            'horizontal':0,
            # ws or up down
            'vertical':0,
            'key_pressed':set(),
            'key_down':set()
        }
        self.inputs_queue = [self.inputs]

    async def listen(self):
        while True:
            event = await self.server.recv(self.reader)
            if not event:
                break
            self.process(event)

    def process(self,event):
        if event['type'] != 'inputs':
            return
        self.inputs_queue.append(event['inputs'])

    def process_input(self):
        all_inputs = self.inputs_queue
        current_inputs = all_inputs[-1]
        # trigger events -> union of all inputs
        trigger = set().union(*(input['trigger'] for input in all_inputs))
        self.inputs['jump'] = 'jump' in trigger
        self.inputs['fire'] = 'fire' in trigger
        key_down = set().union(*(input['key_down'] for input in all_inputs))
        self.inputs['key_down'] = key_down
        # continuous events -> last input
        self.inputs['mouse_pos'] = np.array(current_inputs['mouse_pos'])
        self.inputs['key_pressed'] = current_inputs['key_pressed']
        self.inputs['horizontal'] = current_inputs['horizontal']
        self.inputs['vertical'] = current_inputs['vertical']
        # clear inputs queue
        current_inputs['key_down'] = set()
        current_inputs['jump'] = False
        current_inputs['fire'] = False
        self.inputs_queue = [current_inputs]

    def get_mouse_pos(self):
        return self.inputs['mouse_pos']
    def get_jump(self):
        return self.inputs['jump']
    def get_fire(self):
        return self.inputs['fire']
    def get_horizontal(self):
        return self.inputs['horizontal']
    def get_vertical(self):
        return self.inputs['vertical']
    def get_key_down(self,key):
        return key in self.inputs['key_down']
    def get_key_pressed(self,key):
        return key in self.inputs['key_pressed']
    
class Game:
    def __init__(self) -> None:
        pygame.init()
        # self.clock = pygame.time.Clock()
        # self.fpstext = FPS(self.clock)
        self.running = True

        x, y = sp.symbols('x y')

        # region sphere
        # self.width, self.height = 4, 8
        # sphere_metric = [[1,0],[0,sp.sin(x)**2]]
        # self.map = Map(x, y, sp.Matrix(sphere_metric).tolist())
        # self.player = Player(pos=np.array([1,1], dtype=np.float64),
        #                      vel=np.array([0, 0], dtype=np.float64),
        #                      speed=2,
        #                      bullet_speed=0.2,
        #                      bullet_radius=0.1,
        #                      bullet_lifetime=5,
        #                      map=self.map,
        #                      color=(255, 255, 255),
        #                      radius=0.2)
        # endregion
        
        # region Poincaré half-plane
        # self.width, self.height = 5, 5
        # half_plane_metric = [[1/y**2,0],[0,1/y**2]]
        # self.map = Map(x,y,sp.Matrix(half_plane_metric).tolist())
        # self.player = Player(pos=np.array([1,1], dtype=np.float64),
        #                      vel=np.array([0, 0], dtype=np.float64),
        #                      speed=1,
        #                      bullet_speed=2,
        #                      bullet_radius=0.1,
        #                      bullet_lifetime=10,
        #                      map=self.map,
        #                      color=(255, 255, 255),
        #                      radius=0.2)
        # endregion

        # region Poincaré disk
        self.width, self.height = 1, 1
        disk_metric =sp.eye(2)*(4/(1-(x**2+y**2))**2)
        self.map = Map(x,y,disk_metric.tolist())
        # self.player = Player(pos=np.array([0,0], dtype=np.float64),
        #                      vel=np.array([0, 0], dtype=np.float64),
        #                      speed=0.2,
        #                      bullet_speed=0.1,
        #                      bullet_radius=0.1,
        #                      bullet_lifetime=20,
        #                      map=self.map,
        #                      color=(255, 255, 255),
        #                      radius=0.2)
        # endregion

        # region Torus
        # R,r=3,2
        # self.width, self.height = 2*sp.pi, 2*sp.pi
        # torus_metric=[[r**2, 0], [0, (R + r*sp.cos(x))**2]]
        # self.map = Map(x,y,sp.Matrix(torus_metric).tolist())
        # self.player = Player(pos=np.array([1, 1], dtype=np.float64),
        #                      vel=np.array([0, 0], dtype=np.float64),
        #                      speed=2,
        #                      bullet_speed=4,
        #                      bullet_radius=0.1,
        #                      bullet_lifetime=20,
        #                      map=self.map,
        #                      color=(255, 255, 255),
        #                     radius=0.2)
        # endregion

        # region Klein bottle
        # self.width, self.height = 1,1
        # klein_bottle_metric=[[1,0],[0,-1]]
        # self.map = Map(x,y,sp.Matrix(klein_bottle_metric).tolist())
        # self.player = Player(pos=np.array([2, 1], dtype=np.float64),
        #                      vel=np.array([0, 0], dtype=np.float64),
        #                      speed=2,
        #                      bullet_speed=4,
        #                      bullet_radius=0.1,
        #                      bullet_lifetime=5,
        #                      map=self.map,
        #                      color=(255, 255, 255),
        #                     radius=0.2)
        # endregion
        
        # 300 bullets -> fps 60
        self.max_bullets = 1000
        self.bullets: list[Bullet] = []

        # self.camera = Camera(
        #     pos=np.array([self.width/2,self.height/2],dtype=np.float64),
        #     width=800,
        #     height=600,
        #     zoom=100
        # )

        self.fps = 120

        self.star_num = 100
        self.star_pos_list = np.random.uniform(0, [self.width,self.height], (self.star_num, 2))
        self.star_size_list = np.random.uniform(0.00, 0.02, self.star_num)
        # self.stars = []
        # for i in range(self.star_num):
        #     x = np.random.uniform(0, self.width)
        #     y = np.random.uniform(0, self.height)
        #     pos = np.array([x,y],dtype=np.float64)
        #     size = np.random.uniform(0, 0.02)
        #     self.stars.append((pos,size))
        # self.map_drawer = MapDrawer(
        #     map=self.map,
        #     step=20,
        #     space=0.1,
        #     length=1,
        #     color=GRAY
        # )
        # self.map_drawer.generate(np.array([1,1]),np.array([1,0]))
        # self.map_drawer.generate(np.random.randn(2),np.random.randn(2))
        # self.map_drawer.generate(np.random.randn(2),np.random.randn(2))
        # self.map_drawer.generate(np.random.randn(2),np.random.randn(2))

        # self.mouse_circle = MouseCircle(
        #     camera=self.camera,
        #     map=self.map,
        #     player=self.player,
        #     pos=np.array([0,0],dtype=np.float64),
        #     radius=0.1,
        #     original_color=GREEN,
        #     collision_color=RED
        # )

        self.server = Server('localhost',8888,self)
    
        self.players:list[Player] = []
        self.cameras:list[Camera] = []
        self.controllers:list[NetworkControllerReader] = []

    def add_player(self,controller:'NetworkControllerReader'):
        random_color = np.random.randint(0,255,(3,))
        player = Player(
            pos=np.array([0.5,0.5], dtype=np.float64),
            vel=np.array([0, 0], dtype=np.float64),
            speed=1,
            bullet_speed=2,
            bullet_radius=0.1,
            bullet_lifetime=10,
            map=self.map,
            color=random_color,
            radius=0.2,
            controller=controller,
            game=self
        )
        self.players.append(player)
        return player
    
    def add_camera(self,player:'Player',
                   screen:NetworkSurfaceWriter,
                   controller:'NetworkControllerReader'):
        camera = Camera(
            pos=np.array([self.width/2,self.height/2],dtype=np.float64),
            width=800,
            height=600,
            zoom=100,
            screen=screen,
            player=player,
            controller=controller
        )
        self.cameras.append(camera)
        return camera

    def add_bullet(self,bullet:'Bullet'):
        self.bullets.append(bullet)

    def add_controller(self,controller:'NetworkControllerReader'):
        self.controllers.append(controller)

    def remove_player(self,player:'Player'):
        self.players.remove(player)

    def remove_camera(self,camera:'Camera'):
        self.cameras.remove(camera)

    def remove_controller(self,controller:'NetworkControllerReader'):
        self.controllers.remove(controller)

    def update(self,delta_time):
        for player in self.players:
            player.update(delta_time)
        
        if len(self.bullets) > self.max_bullets:
            self.bullets=self.bullets[len(self.bullets)-self.max_bullets:]
        
        to_del=[]
        for bullet in self.bullets:
            bullet.update(delta_time)

            for player in self.players:
                bullet.check_collision(player)

            if not bullet.alive:
                to_del.append(bullet)

        for bullet in to_del:
            self.bullets.remove(bullet)

        for camera in self.cameras:
            camera.update()
        
    def draw(self):

        for camera in self.cameras:
            camera.clear()

            # camera.draw_circles((200,200,200),self.star_pos_list,self.star_size_list)
            
            for player in self.players:
                player.draw(camera)

            for bullet in self.bullets:
                bullet.draw(camera)

            camera.draw()

    def process_input(self):
        for controller in self.controllers:
            controller.process_input()

    async def tick(self):
        delta_time = time.time()-self.current_time
        self.current_time = time.time()
        await asyncio.sleep(1/self.fps-delta_time)
        return delta_time

    async def run(self):
        await self.server.start_server()
        self.current_time = time.time()
        while self.running:

            delta_time = await self.tick()
            
            self.process_input()

            self.update(delta_time)

            self.draw()

            await self.server.drain()

        await self.server.close()
        pygame.quit()
        # self.server.close()


if __name__ == '__main__':
    game = Game()
    asyncio.run(game.run(),debug=True)

    # import cProfile
    # import pstats
    # with cProfile.Profile() as pr:
    #     game = Game()
    #     game.run()
    # stats = pstats.Stats(pr)
    # stats.sort_stats(pstats.SortKey.TIME)
    # # stats.print_stats()
    # # show use snakeviz
    # stats.dump_stats("profile.prof")
    # # snakeviz profile.prof


    # from scalene import scalene_profiler

    # # Turn profiling on
    # scalene_profiler.start()

    # game = Game()
    # game.run()
    # # Turn profiling off
    # scalene_profiler.stop()
    