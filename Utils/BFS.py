import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from Models.data_obj import Node, Edge, ZONE_COLORS, COLORS

class BFS:
    def __init__(self, graph, start_node):
        self.graph = graph
        self.start_node = start_node
        self.visited = set()
        self.queue = deque([start_node])
        self.path = []

    def search(self):
        while self.queue:
            current_node = self.queue.popleft()
            if current_node not in self.visited:
                self.visited.add(current_node)
                self.path.append(current_node)
                for edge in self.graph.get_edges(current_node):
                    neighbor = edge.target if edge.source == current_node else edge.source
                    if neighbor not in self.visited:
                        self.queue.append(neighbor)
        self.path

    def visualize(self):
        fig, ax = plt.subplots()

        # Plot nodes
        for node in self.graph.nodes:
            ax.scatter(node.pos[0], node.pos[1],
                    c=ZONE_COLORS.get(node.zone, 'r'),
                    s=100)
            ax.text(node.pos[0], node.pos[1], str(node.id))

        # Plot graph edges (original graph edges)
        for edge in self.graph.edges:
            src_pos = edge.source.pos
            dest_pos = edge.target.pos

            # Convert color to matplotlib compatible format
            color = tuple(c/255 for c in edge.color[:3])

            ax.plot([src_pos[0], dest_pos[0]],
                    [src_pos[1], dest_pos[1]],
                    color=color,
                    linewidth=2,
                    alpha=0.3)  # Make original edges slightly transparent

        # Highlight BFS path
        if len(self.path) > 1:
            bfs_path_x = [node.pos[0] for node in self.path]
            bfs_path_y = [node.pos[1] for node in self.path]

            # Plot BFS path
            ax.plot(bfs_path_x, bfs_path_y,
                    color='blue',
                    linewidth=4,
                    linestyle='--',
                    label='BFS Path')

            # Highlight start and end nodes of BFS
            ax.scatter(self.path[0].pos[0], self.path[0].pos[1],
                    color='green',
                    s=200,
                    marker='o',
                    edgecolors='black',
                    label='Start Node')
            ax.scatter(self.path[-1].pos[0], self.path[-1].pos[1],
                    color='red',
                    s=200,
                    marker='o',
                    edgecolors='black',
                    label='End Node')

        plt.title('BFS Traversal')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.legend()
        plt.show()
