import numpy as np

class Node:
    def __init__(self, node_id, pos):
        self.id = node_id
        self.pos = np.array(pos)


class Edge:
    def __init__(self, source, target, color=(0, 0, 0, 255)):
        self.source = source  # Node object
        self.target = target  # Node object
        self.color = color