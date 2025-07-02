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
    def __init__(self, node1, node2, weight, capacity, color=(0, 0, 0, 255), passengers=None):
        # Store nodes as a frozenset for undirected logic
        self._nodes = frozenset({node1, node2})

        # Keep source/target as they were, but normalize based on ID to make it consistent
        self._source, self._target = sorted([node1, node2], key=lambda n: n.id)

        self.color = color
        self.weight = weight
        self.capacity = capacity
        self.passengers = passengers if passengers is not None else set()

    @property
    def source(self):
        return self._source

    @property
    def target(self):
        return self._target

    def __repr__(self):
        return f"{self.source.id} <-> {self.target.id}"

    def __hash__(self):
        # Order-independent hash
        return hash(self._nodes)

    def __eq__(self, other):
        return isinstance(other, Edge) and self._nodes == other._nodes

    def __gt__(self, other):
        return self.weight > other.weight

    def __lt__(self, other):
        return self.weight < other.weight

    @property
    def nodes(self):
        return tuple(self._nodes)

    @property
    def SetTraficColor(self):
        occupancy_ratio = len(self.passengers) / self.capacity
        if occupancy_ratio < 0.3:
            self.color = (0, 200, 0, 255)
        elif occupancy_ratio < 0.6:
            self.color = (255, 255, 0, 255)
        elif occupancy_ratio < 0.8:
            self.color = (255, 0, 0, 255)
        elif occupancy_ratio < 1:
            self.color = (128, 0, 0, 255)
        else:
            self.color = (0, 0, 0, 255)


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
