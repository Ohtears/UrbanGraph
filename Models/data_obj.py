import numpy as np

class Node:
    def __init__(self, node_id, pos, zone=None):
        self.id = node_id
        self.pos = np.array(pos)
        self.zone = zone

class Edge:
    def __init__(self, source, target, color=(0, 0, 0, 255)):
        self.source = source  # Node object
        self.target = target  # Node object
        self.color = color

ZONE_COLORS = {
    0: 'c',  # cyan
    1: 'm',  # magenta
    2: 'y',  # yellow
    3: 'g',  # green
}

COLORS = {
    "c" : (0, 255, 255, 255),  # cyan
    "m" : (255, 0, 255, 255),  # magenta
    "y" : (255, 255, 0, 255),  # yellow
    "g" : (0, 255, 0, 255),  # green
}