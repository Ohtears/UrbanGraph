import numpy as np
import heapq
from Models.data_obj import Node, Edge, ZONE_COLORS, COLORS

class PrimMST:
    def __init__(self):
        """
        Initialize Prim's MST with Node and Edge objects

        :param nodes: List of Node objects
        :param edges: List of Edge objects
        """

        self.mst_nodes = set()
        self.mst_edges = []
        self.num_nodes = 0

    def compute_mst(self, nodes, edges):
        """
        Compute Minimum Spanning Tree using Prim's algorithm

        :return: List of Edge objects in the Minimum Spanning Tree
        """
        # Create adjacency list representation
        graph = {}
        for node in nodes:
            graph[node] = []

        # Populate graph with edges
        for edge in edges:
            graph[edge.source].append((edge.target, edge.weight))
            graph[edge.target].append((edge.source, edge.weight))

        # Track visited nodes
        visited = set()

        # Minimum Heap to store edges
        min_heap = []

        # Result to store MST
        mst_edges = []

        # Start from the first node
        start_node = nodes[0]
        visited.add(start_node)

        # Add all edges from the start node to the heap
        for neighbor, weight in graph[start_node]:
            heapq.heappush(min_heap, (weight, start_node, neighbor))

        # Continue until all nodes are visited
        while min_heap:
            weight, src, dest = heapq.heappop(min_heap)

            # Skip if destination is already visited
            if dest in visited:
                continue

            # Mark destination as visited
            visited.add(dest)

            # Create and add edge to MST
            # Use the original edge color or a default
            edge_color = None
            for original_edge in edges:
                if (original_edge.source == src and original_edge.target == dest) or \
                   (original_edge.source == dest and original_edge.target == src):
                    edge_color = original_edge.color
                    break

            # If no original edge found, use a default color
            if edge_color is None:
                edge_color = (0, 0, 0, 255)  # Default black

            mst_edge = Edge(src, dest, weight, color=edge_color)
            mst_edges.append(mst_edge)

            # Add new edges from the current node
            for next_neighbor, next_weight in graph[dest]:
                if next_neighbor not in visited:
                    heapq.heappush(min_heap, (next_weight, dest, next_neighbor))

        self.mst_edges = mst_edges

    def get_mst_total_weight(self):
        """
        Calculate total weight of the Minimum Spanning Tree

        :param mst_edges: List of Edge objects in the MST
        :return: Total weight of the MST
        """
        return sum(edge.weight for edge in self.mst_edges)


    def visualize_mst(self, nodes, ax=None):

        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        # Plot nodes
        for node in nodes:
            ax.scatter(node.pos[0], node.pos[1],
                    c=ZONE_COLORS.get(node.zone, 'r'),
                    s=100)
            ax.text(node.pos[0], node.pos[1], str(node.id))

        # Plot MST edges
        for edge in self.mst_edges:
            src_pos = edge.source.pos
            dest_pos = edge.target.pos

            # Convert color to matplotlib compatible format
            color = tuple(c/255 for c in edge.color[:3])

            ax.plot([src_pos[0], dest_pos[0]],
                    [src_pos[1], dest_pos[1]],
                    color=color,
                    linewidth=2)

        ax.set_title("Minimum Spanning Tree")
        plt.show()

    def get_mst_nodes(self):

        mst_nodes = set()
        for edge in self.mst_edges:
            mst_nodes.add(edge.source)
            mst_nodes.add(edge.target)

        self.mst_nodes = mst_nodes
        return mst_nodes

    def filter_mst_by_zone(self, target_zone):
        """
        Filter MST edges by a specific zone

        :param mst_edges: List of Edge objects in the MST
        :param target_zone: Zone to filter by
        :return: List of edges in the specified zone
        """
        return [
            edge for edge in self.mst_edges
            if edge.source.zone == target_zone or edge.target.zone == target_zone
        ]

    def add_node(self, node, edges) :
        self.mst_nodes.add(node)
        min_edge = min(edges)
        self.mst_edges.append(min_edge)
