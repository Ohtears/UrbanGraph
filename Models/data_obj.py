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
        self.passengers = passengers if passengers is not None else Queue()

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

    def set_traffic_color(self):
        occupancy_ratio = len(self.passengers) / self.capacity

        if len(self.passengers) == 0:
            self.color = (0, 200, 0, 255) # Green for no passengers
        elif len(self.passengers) == 1:
            self.color = (255, 255, 0, 255) # Yellow for 1 passenger
        elif len(self.passengers) == 2:
            self.color = (255, 0, 0, 255) # Red for 2 passengers
        elif len(self.passengers) == 3:
            self.color = (128, 0, 0, 255) # Dark red for 3 passengers
        else:
            self.color = (0, 0, 0, 255) # Black for more than 3 passengers or full capacity


class Graph:
    def __init__(self,nodes, edges):
        self.graph = {}
        self.nodes = nodes
        self.edges = edges
        for node in nodes:
            self.graph[node.id] = []

        # Populate graph with edges
        for edge in edges:
            self.graph[edge.source.id].append(edge)
            self.graph[edge.target.id].append(edge)

    def get_edges(self, node):
        try :
            return self.graph.get(node.id, [])
        except :
            return self.graph.get(node, [])


class Queue :
    def __init__(self):
        self.queue = []
    def push(self, value) :
        self.queue.append(value)

    def pop(self) :
        value = self.queue[0]
        self.queue = self.queue[1:]
        return value

    def __len__(self):
        return len(self.queue)

ZONE_COLORS = {
    0: 'c',      # cyan
    1: 'm',      # magenta
    2: 'y',      # yellow
    3: 'b',      # blue 
}

COLORS = {
    "c" : (0, 255, 255, 255),  # cyan
    "m" : (255, 0, 255, 255),  # magenta
    "y" : (255, 255, 0, 255),  # yellow
    "g" : (0, 0, 255, 255),  # blue
}
