import numpy as np

class TSP:
    def __init__(self, AstarAgent):
        self.graph = AstarAgent.graph
        self.num_nodes = len(AstarAgent.graph.graph)
        self.distances = np.full((self.num_nodes, self.num_nodes), np.inf)
        self.astar_agent = AstarAgent

    @staticmethod
    def unique_preserve_order(seq):
        seen = set()
        result = []
        for item in seq:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result

    def tsp(self, source, destinations):
        source_index = source.id
        dest_indices = [d.id for d in destinations]
        # build a lookup so we can ask A* for a given ID later:
        id2node = {source.id: source}
        for d in destinations:
            id2node[d.id] = d

        n = len(destinations)
        ALL = (1 << n) - 1

        # 1) Seed distances from source → dest[i]
        for i, dest in enumerate(destinations):
            path, edges = self.astar_agent.a_star_search(source, dest)
            if path is not None:
                cost = sum(e.weight for e in edges)
                self.distances[source_index][dest.id] = cost
                # print(f"dist[{source_index}→{dest.id}] = {cost}")

        # 2) Seed distances dest[i] → dest[j]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                u = destinations[i]
                v = destinations[j]
                path, edges = self.astar_agent.a_star_search(u, v)
                if path is not None:
                    cost = sum(e.weight for e in edges)
                    self.distances[u.id][v.id] = cost

        # 3) DP table
        dp = np.full((1 << n, n), np.inf)
        for i in range(n):
            dp[1 << i][i] = self.distances[source_index][dest_indices[i]]

        for mask in range(1 << n):
            for u in range(n):
                if not (mask & (1 << u)):
                    continue
                for v in range(n):
                    if mask & (1 << v):
                        continue
                    nxt = mask | (1 << v)
                    new_cost = dp[mask][u] + self.distances[dest_indices[u]][dest_indices[v]]
                    if new_cost < dp[nxt][v]:
                        dp[nxt][v] = new_cost

        # 4) Extract best end‐node, reconstruct ID‐path
        min_cost = np.inf
        best_path = []
        for i in range(n):
            c = dp[ALL][i]
            if c < min_cost:
                min_cost = c
                best_path = self._reconstruct_path(dp, ALL, i, dest_indices)

        # 5) Re‐derive the chosen edges by replaying A* on the found node sequence
        chosen_edges = []
        chosen_nodes = []
        prev_node = source
        for nid in best_path:
            curr_node = id2node[nid]
            nodes, edges = self.astar_agent.a_star_search(prev_node, curr_node)
            if edges is None:
                raise RuntimeError(f"A* failed on leg {prev_node.id}→{curr_node.id}")
            # append the full list of edges for this leg
            chosen_edges.extend(edges)
            chosen_nodes.extend(nodes)
            prev_node = curr_node

        return min_cost, best_path, self.unique_preserve_order(chosen_nodes), self.unique_preserve_order(chosen_edges)


    def _reconstruct_path(self, dp, mask, last, dest_indices):
        path = []
        n = len(dest_indices)
        while mask:
            path.append(dest_indices[last])
            prev_mask = mask & ~(1 << last)
            for u in range(n):
                if not (prev_mask & (1 << u)):
                    continue
                if dp[prev_mask][u] + self.distances[dest_indices[u]][dest_indices[last]] == dp[mask][last]:
                    last = u
                    break
            mask = prev_mask
        path.reverse()
        return path
