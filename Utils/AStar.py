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
                # Option 1 : dont try the blocked edges !
                # valid_edges = [e for e in connecting_edges if not self.is_edge_blocked(e)]

                # Option 2 : dublle the blocked edges' weight
                valid_edges = connecting_edges
                if not valid_edges:
                    continue

                edge = valid_edges[0]  # You can improve edge selection if needed
                trafick_cost = edge.weight if self.is_edge_blocked(edge) else 0
                tentative_g_score = g_score[current_node] + edge.weight + trafick_cost

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current_node
                    edge_used[(current_node, neighbor)] = edge
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal_node)

                    if neighbor not in open_set_entries:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_entries.add(neighbor)

        return None, None  # No path found

    @staticmethod
    def filter_border_by_zone(border_dict, zone):
        zone_str = str(zone)
        result = {}

        for key, value in border_dict.items():
            if zone_str in key.split("to"):
                result[key] = value

        return result

    def performA_star(self, start_node, goal_node, border_nodes, log=False):
        node1_zone = start_node.zone
        node2_zone = goal_node.zone

        if node1_zone == node2_zone:
            return self.a_star_search(start_node, goal_node)

        key = f"{min(node1_zone, node2_zone)}to{max(node1_zone, node2_zone)}"
        best_cost = np.inf
        best_path, best_edges = None, None

        if border_nodes.get(key):
            candidates = border_nodes[key]
        else:
            # Fallback to indirect borders
            trimed_dict = self.filter_border_by_zone(node1_zone, border_nodes)
            candidates = [node for nodes in trimed_dict.values() if nodes for node in nodes]

        for border_node in candidates:
            path1, edges1 = self.a_star_search(start_node, border_node)
            path2, edges2 = self.a_star_search(border_node, goal_node)

            if not path1 or not path2:
                continue

            cost = sum(e.weight for e in edges1 + edges2)

            if cost < best_cost:
                if log :
                    print(f"New best cost is {cost}")
                    print(f"Prev best path : {best_path}")
                    print(f"Prev best path : {best_edges}")

                best_cost = cost
                best_path = path1[:-1] + path2
                best_edges = edges1 + edges2
                # print(border_node)

        return best_path, best_edges
