import numpy as np

class Node:
    def __init__(self, node_id, pos, zone=None):
        self.id = node_id
        self.pos = np.array(pos)
        self.zone = zone

    def __gt__(self, oprand) :
        return  self.zone < oprand.zone

    def __lt__(self, oprand) :
        return  self.zone > oprand.zone
    def __repr__(self):
        return str(self.id)

class Edge:
    def __init__(self, source, target, weight, capacity, color=(0, 0, 0, 255), passengers=0):
        self.source = source  # Node object
        self.target = target  # Node object
        self.color = color
        self.weight = weight
        self.capacity = capacity
        self.passengers = passengers

    def __gt__(self, oprand) :
        return self.weight > oprand.weight
    def __lt__(self, oprand) :
        return self.weight < oprand.weight
    def __eq__(self, oprand) :
        return self.weight == oprand.weight

    def __repr__(self):
        return f"{self.source.id} -> {self.target.id}"

    @property
    def SetTraficColor(self):
        occupancy_ratio = self.passengers / self.capacity

        if occupancy_ratio < 0.3:
            self.color = (0, 200, 0, 255)  # Dark green
        elif occupancy_ratio < 0.6:
            self.color = (255, 255, 0, 255)  # Yellow
        elif occupancy_ratio < 0.8:
            self.color = (255, 0, 0, 255)  # Red
        elif occupancy_ratio < 1:
            self.color = (128, 0, 0, 255)  # Dark red
        else:
            self.color = (0, 0, 0, 255)  # Black

class Graph:
    def __init__(self,nodes, edges):
        self.graph = {}
        for node in nodes:
            self.graph[node.id] = []

        # Populate graph with edges
        for edge in edges:
            self.graph[edge.source.id].append(edge)
            self.graph[edge.target.id].append(edge)

    def get_edges(self, node):
        return self.graph.get(node.id, [])

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
