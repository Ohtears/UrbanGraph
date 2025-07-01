import heapq
import numpy as np
from Models.data_obj import Node, Edge, Graph, ZONE_COLORS, COLORS

class AStar:
    def __init__(self, nodes, edges):
        self.graph = Graph(nodes, edges)

    def heuristic(self, start_node, goal_node):
        geometric_distance = np.linalg.norm(start_node.pos - goal_node.pos)
        zone_distance = abs(start_node.zone - goal_node.zone)
        return (0.7 * geometric_distance) + (0.3 * zone_distance * 10)

    def is_edge_blocked(self, edge):
        return len(edge.passengers) >= edge.capacity

    def get_neighbor_edges(self, node):
        return self.graph.get_edges(node)

    def get_neighbor_nodes(self, node):
        neighbor_edges = self.get_neighbor_edges(node)
        neighbors = set()
        for edge in neighbor_edges:
            if edge.source == node:
                neighbors.add(edge.target)
            elif edge.target == node:
                neighbors.add(edge.source)
        return list(neighbors)

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

    def a_star_search(self, start_node, goal_node):
        open_set = []
        open_set_entries = set()
        heapq.heappush(open_set, (0, start_node))
        open_set_entries.add(start_node)

        came_from = {}
        edge_used = {}

        g_score = {start_node: 0}
        f_score = {start_node: self.heuristic(start_node, goal_node)}
        closed_set = set()

        while open_set:
            current_f, current_node = heapq.heappop(open_set)
            open_set_entries.discard(current_node)

            if current_node == goal_node:
                node_path = self.reconstruct_path(came_from, current_node)
                edge_path = []
                for i in range(len(node_path) - 1):
                    edge_path.append(edge_used[(node_path[i], node_path[i + 1])])
                return node_path, edge_path

            closed_set.add(current_node)

            for neighbor in self.get_neighbor_nodes(current_node):
                if neighbor in closed_set:
                    continue

                # Find valid connecting edge
                connecting_edges = [
                    edge for edge in self.get_neighbor_edges(current_node)
                    if ((edge.source == current_node and edge.target == neighbor) or
                        (edge.target == current_node and edge.source == neighbor))
                ]
                valid_edges = [e for e in connecting_edges if not self.is_edge_blocked(e)]
                if not valid_edges:
                    continue

                edge = valid_edges[0]  # You can improve edge selection if needed
                tentative_g_score = g_score[current_node] + edge.weight

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current_node
                    edge_used[(current_node, neighbor)] = edge
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal_node)

                    if neighbor not in open_set_entries:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_entries.add(neighbor)

        return None, None  # No path found
