#
# https://stackoverflow.com/questions/21019471/how-can-i-draw-a-3d-shape-using-pygame-no-other-modules
#
import pygame
from numpy import array
from math import cos, sin
from pygame import K_q, K_w, K_a, K_s, K_z, K_x, K_ESCAPE, K_1


######################
#                    #
#    math section    #
#                    #
######################


class Physical:
    def __init__(self, vertices, edges, _rot=[0.2, 0.3, -0.06]):
        """
        a 3D object that can rotate around the three axes
        :param vertices: a tuple of points (each has 3 coordinates)
        :param edges: a tuple of pairs (each pair is a set containing 2 vertices' indexes)
        """
        self.__vertices = array(vertices)
        self.__edges = tuple(edges)
        self.__rotation = _rot  #[0, 0, 0]  # radians around each axis

    def rotate(self, axis, θ):
        self.__rotation[axis] += θ

    def _rotation_matrix(self, α, β, γ):
        """
        rotation matrix of α, β, γ radians around x, y, z axes (respectively)
        """
        sα, cα = sin(α), cos(α)
        sβ, cβ = sin(β), cos(β)
        sγ, cγ = sin(γ), cos(γ)
        return (
            (cβ * cγ, -cβ * sγ, sβ),
            (cα * sγ + sα * sβ * cγ, cα * cγ - sγ * sα * sβ, -cβ * sα),
            (sγ * sα - cα * sβ * cγ, cα * sγ * sβ + sα * cγ, cα * cβ)
        )

    @property
    def lines(self):
        location = self.__vertices.dot(self._rotation_matrix(*self.__rotation))  # an index->location mapping
        return ((location[v1], location[v2]) for v1, v2 in self.__edges)


######################
#                    #
#    gui section     #
#                    #
######################

class Paint:
    def __init__(self, shape, keys_handler):
        self.LGREY = (222, 222, 222)
        self.DGREY = (111, 111, 111)
        self.__shape = shape
        self.__keys_handler = keys_handler
        #self.__size = 1920, 1080
        self.__size = 800, 600
        #self.__size = 0, 0
        self.__clock = pygame.time.Clock()
        #self.__screen = pygame.display.set_mode(self.__size, pygame.FULLSCREEN)
        self.__screen = pygame.display.set_mode(self.__size)

    def run(self):
        self.__mainloop()


    def update(self, shape):
        self.__shape = shape
        self.__screen.fill(self.LGREY)
        self.__draw_shape()

    def __fit(self, vec):
        """
        ignore the z-element (creating a very cheap projection), and scale x, y to the coordinates of the screen
        """
        # notice that len(self.__size) is 2, hence zip(vec, self.__size) ignores the vector's last coordinate
        return [round(40 * coordinate + frame / 2) for coordinate, frame in zip(vec, self.__size)]

    def __handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
        self.__keys_handler(pygame.key.get_pressed())

    def __draw_shape(self, thickness=4):
        for start, end in self.__shape.lines:
            pygame.draw.line(self.__screen, self.DGREY, self.__fit(start), self.__fit(end), thickness)

    def __mainloop(self):
        while True:
            self.__handle_events()
            self.__screen.fill(self.LGREY)
            self.__draw_shape()
            pygame.display.flip()
            self.__clock.tick(40)


######################
#                    #
#     main start     #
#                    #
######################

class GameDemo:
    def __init__(self):
        self.X, self.Y, self.Z = 0, 1, 2
        self.cube = Physical(
            vertices=((0, 0, 0), (0, 0, 0)),
            edges=({0, 1}, {1, 0})
        )
        self.counter_clockwise = 0.05  # radians
        self.clockwise = -self.counter_clockwise
        self.params_cam = {
            K_q: (self.X, self.clockwise),
            K_w: (self.X, self.counter_clockwise),
            K_a: (self.Y, self.clockwise),
            K_s: (self.Y, self.counter_clockwise),
            K_z: (self.Z, self.clockwise),
            K_x: (self.Z, self.counter_clockwise),
        }
        self.painter = Paint(self.cube, self.keys_handler)
        self.worker_opt = True
        self.vertives_ind = [15, 14, 13, 8, 10, 11, 12]
        self.edges_ref = ({0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6})

    def keys_handler(self, keys):
        for key in self.params_cam:
            if keys[key]:
                self.cube.rotate(*self.params_cam[key])

        if keys[K_ESCAPE]:
            self.worker_opt = False
            exit()

        if keys[K_1]:
            self._update_pose_test()            

    def run(self):
        pygame.init()
        pygame.display.set_caption('Control -   q,w : X    a,s : Y    z,x : Z')
        self.painter.run()

    def _update_pose_test(self):
        self.cube = Physical(
            vertices=(
                (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1), (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)),
            edges=({0, 1}, {0, 2}, {2, 3}, {1, 3},
                   {4, 5}, {4, 6}, {6, 7}, {5, 7},
                   {0, 4}, {1, 5}, {2, 6}, {3, 7}),
            #rot=self.cube.__rotation
        )
        self.painter.update(self.cube)

    def _get_poi(self, tri_di_pose):
        vertices_l = [((-1)*tri_di_pose[0][ii, 0]*(7), tri_di_pose[0][ii, 1]*(7), tri_di_pose[0][ii, 2]*(7)) for ii in self.vertives_ind]        
        return vertices_l

    def draw_pose(self, tri_di_pose):
        self.cube = Physical(
            vertices=self._get_poi(tri_di_pose),
            edges=self.edges_ref
        )

        self.painter.update(self.cube)