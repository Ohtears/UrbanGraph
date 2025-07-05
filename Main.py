import sys
import os
from matplotlib.typing import HashableList
import numpy as np
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QGraphicsPixmapItem
import pyqtgraph as pg
import networkx as nx
import random
import threading
import time
from datetime import datetime
from PyQt6.QtCore import QTimer
from collections import deque
import heapq
from collections import defaultdict
from itertools import combinations

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Models.data_obj import Node, Edge, Graph, ZONE_COLORS, COLORS
from Models.User import User
from Utils.MST import PrimMST
from Utils.AStar import AStar
from Utils.Clock import ClockGenerator
from Utils.TSP import TSP
from Utils.BFS import BFS


import builtins

def print_with_timestamp(*args, **kwargs):
    now = datetime.now().strftime("[%H:%M:%S.%f]")
    builtins._original_print(now, *args, **kwargs)

builtins._original_print = builtins.print
builtins.print = print_with_timestamp


class GraphApp(QWidget):
    def __init__(self, clock):
        super().__init__()
        self.clock = clock
        self.tsp_queue = []
        self.tsp_working = False
        self.active_users = []
        self._clock_tick_thread = threading.Thread(
            target=self._tick_loop,
            daemon=True
        )
        self._clock_tick_thread.start()

        bg_pixmap = QPixmap("BackgroundImg.png")

        self.arrow_timers = {}  # Dictionary to track arrows and their corresponding timers

        self.setWindowTitle("City")
        self.graph_widget = pg.GraphicsLayoutWidget()
        self.view = self.graph_widget.addViewBox()
        self.view.setAspectLocked()
        self.view.setMouseMode(self.view.PanMode)
        self.view.scene().sigMouseClicked.connect(self.on_click)

        bg_item = QGraphicsPixmapItem(bg_pixmap)
        bg_item.setZValue(-10)

        bg_rect = bg_pixmap.rect()
        bg_item.setTransformOriginPoint(bg_rect.width() / 2, bg_rect.height() / 2)
        bg_item.setScale(0.2)

        bg_item.setPos(-bg_rect.width() / 2, -bg_rect.height() / 2)

        self.view.addItem(bg_item)
        self.graph_item = pg.GraphItem()
        self.view.addItem(self.graph_item)

        self.nodes = []
        self.edges = []
        self.border_nodes = defaultdict()
        self.pos = np.array([])

        self.mst_solver = PrimMST()

        layout = QVBoxLayout()
        layout.addWidget(self.graph_widget)

        self.saved_traffic_colors = {}
        self.flow_arrows = []
        self.travel_mode = False
        self.travel_path = []

        self.travel_btn = QPushButton("Travel")
        self.travel_btn.clicked.connect(self.start_travel)
        layout.addWidget(self.travel_btn)

        self.done_btn = QPushButton("Done")
        self.done_btn.clicked.connect(self.finish_travel)
        self.done_btn.hide()
        layout.addWidget(self.done_btn)

        self.exit_travel_mode_btn = QPushButton("exit_travel_mode")
        self.exit_travel_mode_btn.clicked.connect(self.exit_travel)
        self.exit_travel_mode_btn.hide()
        layout.addWidget(self.exit_travel_mode_btn)

        self.traffic_btn = QPushButton("Toggle Traffic Colors")
        self.traffic_btn.setCheckable(True)
        self.traffic_btn.clicked.connect(self.toggle_traffic)
        layout.addWidget(self.traffic_btn)

        self.bfs_btn = QPushButton("BFS")
        self.bfs_btn.clicked.connect(self.start_bfs)
        layout.addWidget(self.bfs_btn)
        self.bfs_mode = False

        self.bfs_done_btn = QPushButton("Show BFS")
        self.bfs_done_btn.clicked.connect(self.finish_bfs)
        self.bfs_done_btn.hide()
        layout.addWidget(self.bfs_done_btn)

        self.add_node_btn = QPushButton("Add Node")
        self.add_node_btn.clicked.connect(self.add_node)
        layout.addWidget(self.add_node_btn)

        self.generate_btn = QPushButton("Generate Random Map")
        self.generate_btn.clicked.connect(self.generate_map)
        layout.addWidget(self.generate_btn)

        self.setLayout(layout)
        self.show_traffic = False
        self.clock_started = False  # Track if clock has started

    def exit_travel(self):
        self.exit_travel_mode_btn.hide()
        self.travel_btn.show()

        self.resume_clock()
        self.stop_flow_animation()

    def pause_clock(self):
        self.clock.stop()

    def resume_clock(self):
        self.clock.start()

    def _tick_loop(self):
        last = -1
        while True:
            current = self.clock.get_clock_value()
            if current != last:
                last = current
                if current == 1:
                    # print("TICK", datetime.now().strftime("%H:%M:%S.%f"))
                    self.tick_users()
                    self.spawn_random_user()
            time.sleep(0.05)

    def toggle_traffic(self):
        self.show_traffic = self.traffic_btn.isChecked()
        print("Traffic visualization:", "ON" if self.show_traffic else "OFF")
        QTimer.singleShot(0, self.update_graph)

    def tick_users(self):
        # Only process users that have completed TSP and are active
        for user in self.active_users:
            user.tick()
            print(user.getLoc())

        # Remove inactive users
        self.active_users = [u for u in self.active_users if u.is_active()]

        if self.show_traffic:
            QTimer.singleShot(0, self.update_graph)

    def spawn_random_user(self):
        if len(self.nodes) < 2:
            return

        for _ in range(2):
            threading.Thread(target=self._spawn_one_user, daemon=True).start()

    def _spawn_one_user(self):
        n = random.randint(2, len(self.nodes)) if len(self.nodes) < 21 else random.randint(2, 21)
        travel_nodes = random.sample(self.nodes, n)
        src_node = travel_nodes[0]
        dst_nodes = travel_nodes[1:]

        # Create user and add to queue instead of handling immediately
        user = User(src_node, priority=2)
        heapq.heappush(self.tsp_queue, (user.priority, user, src_node, dst_nodes, False))

        # Start processing if not already working
        if not self.tsp_working:
            self.process_tsp_queue()

    def process_tsp_queue(self):
        if not self.tsp_queue or self.tsp_working:
            return

        self.tsp_working = True
        _, user, src_node, dst_nodes, log = heapq.heappop(self.tsp_queue)

        # Calculate TSP in a separate thread to avoid blocking UI
        def calculate_tsp():
            astar = AStar(self.nodes, self.edges)
            tsp_solver = TSP(astar)

            if len(dst_nodes) == 1:
                nodes, path = astar.performA_star(src_node, dst_nodes[0], self.border_nodes)
                min_cost = path[-1].weight if path else 0
                best_order = [dst_nodes[0].id]
            else:
                min_cost, best_order, nodes, path = tsp_solver.tsp(src_node, dst_nodes, self.border_nodes)

            # When calculation is done, add to active users
            user.set_route(nodes, path)
            self.active_users.append(user)

            if log and len(dst_nodes) > 1:
                print(f"User{user.id} starting at {src_node.id} to {[d.id for d in dst_nodes]}")
                print(f"the minimum cost is {min_cost}")
                print(f"best order of destinations {best_order}")

            # Mark TSP as done and process next in queue
            self.tsp_working = False
            QTimer.singleShot(0, self.process_tsp_queue)  # Process next in queue

        # Start calculation in a thread
        threading.Thread(target=calculate_tsp, daemon=True).start()

    def start_bfs(self):
            self.bfs_mode = True
            self.bfs_node = None
            self.bfs_btn.hide()
            self.bfs_done_btn.show()
            print("BFS mode activated. Click on nodes to record your node.")

    def finish_bfs(self):
        self.bfs_mode = False
        self.done_btn.hide()
        self.bfs_btn.show()

        self.BFS_manager()

    def start_travel(self):
        self.travel_mode = True
        self.travel_path = []
        self.travel_btn.hide()
        self.done_btn.show()
        print("Travel mode activated. Click on nodes to record your path.")

        self.pause_clock()
        self.save_traffic_colors()

    def save_traffic_colors(self):
        # Save the traffic color for each edge, updating it first
        self.saved_traffic_colors = {}
        for e in self.edges:
            e.set_traffic_color()
            self.saved_traffic_colors[e] = e.color

    def finish_travel(self):
        self.travel_mode = False
        self.done_btn.hide()
        self.exit_travel_mode_btn.show()
        # self.travel_btn.show()
        if len(self.travel_path) == 0 :
            print("Travel has been canceled !")

        elif len(self.travel_path) > 1 and len(self.travel_path) <= len(self.nodes) :
            dsts = [self.nodes[self.travel_path[i]] for i in range(1, len(self.travel_path))]
            self.Handle_travel(self.nodes[self.travel_path[0]], dsts, log=True, is_user=True)

    def start_flow_animation(self, nodes):
        # Create a single arrow at the start of the first node in the path
        self.flow_arrow = self.create_flow_arrow(nodes[0])  # Start the arrow at the first node in the path
        self.view.addItem(self.flow_arrow)

        self.path = nodes  # Store the path (nodes) for animation
        self.current_node_index = 0  # Start from the first node
        self.animation_duration = 3  # Duration for moving across one edge (in seconds)
        self.start_time = time.time()  # Keep track of the start time for animation

        # Start the animation loop
        self.animate_flow_arrow()

    def create_flow_arrow(self, node):
        # Start the arrow at the position of the first node in the path
        start_pos = node.pos
        arrow = pg.ArrowItem(angle=0)  # Create a simple arrow

        # Set the initial position of the arrow
        arrow.setPos(start_pos[0], start_pos[1])
        return arrow

    def animate_flow_arrow(self):
        elapsed_time = time.time() - self.start_time

        total_duration = self.animation_duration * (len(self.path) - 1)
        progress = (elapsed_time % total_duration) / total_duration

        # Current and next node based on current node index
        current_node = self.path[self.current_node_index]
        next_node = self.path[(self.current_node_index + 1) % len(self.path)]

        start_pos = current_node.pos
        end_pos = next_node.pos

        travel_time = elapsed_time % self.animation_duration
        travel_progress = travel_time / self.animation_duration

        # Move the arrow to the new position
        new_x = start_pos[0] + (end_pos[0] - start_pos[0]) * travel_progress
        new_y = start_pos[1] + (end_pos[1] - start_pos[1]) * travel_progress
        self.flow_arrow.setPos(new_x, new_y)
        travel_progress = travel_progress + 0.03
        if travel_progress >= 1:
            if self.current_node_index == len(self.path) - 2:
                self.current_node_index = 0
                self.start_time = time.time()
            else:
                self.current_node_index = (self.current_node_index + 1) % len(self.path)
                self.start_time = time.time()

        QTimer.singleShot(50, self.animate_flow_arrow)

    def stop_flow_animation(self):
        # Stop the flow animation and hide the arrow
        if self.flow_arrow:
            self.flow_arrow.setVisible(False)  # Hide the arrow
            self.flow_arrow = None  # Optionally remove the arrow object

    def Handle_travel(self, src, dsts, log=True, is_user=False):
        u = User(src)

        if is_user:
            # For user-initiated travel, highlight path when done

            # Add to queue with callback
            astar = AStar(self.nodes, self.edges)
            tsp_solver = TSP(astar)

            if len(dsts) == 1:
                nodes, path = astar.performA_star(src, dsts[0], self.border_nodes)
                print(nodes,path)
                min_cost = path[-1].weight if path else 0
                best_order = [dsts[0].id]
            else:
                min_cost, best_order, nodes, path = tsp_solver.tsp(src, dsts, self.border_nodes)

            # When calculation is done, add to active users
            u.set_route(nodes, path)
            self.active_users.append(u)

            if log and len(dsts) > 1:
                print(f"User{u.id} starting at {src.id} to {[d.id for d in dsts]}")
                print(f"the minimum cost is {min_cost}")
                print(f"best order of destinations {best_order}")


            self.highlight_travel_path(u.travel_path, u.travel_nodes)


        else:
            # For random spawns, just add to queue
            # heapq.heappush(self.tsp_queue, (u, src, dsts, True))
            heapq.heappush(self.tsp_queue, (u.priority, u, src, dsts, True))

            if not self.tsp_working:
                self.process_tsp_queue()

    def highlight_travel_path(self, path, nodes):
        print("Highlighting travel path...")

        pens = []

        for edge in self.edges:
            if edge in path:
                pens.append(self.saved_traffic_colors.get(edge))
            else:
                pens.append((0, 0, 0, 0))  # nothing

        print(pens)
        adj = np.array([[e.source.id, e.target.id] for e in self.edges])
        symbols = ['o'] * len(self.nodes)
        sizes = [10] * len(self.nodes)
        brushes = [ZONE_COLORS.get(node.zone, 'w') for node in self.nodes]

        # Update the graph to highlight the path
        self.graph_item.setData(
            pos=self.pos,
            adj=adj,
            size=sizes,
            symbol=symbols,
            pxMode=True,
            pen=np.array(pens),
            brush=brushes
        )

        self.start_flow_animation(nodes)

    def BFS_manager(self) :
        graph = Graph(self.nodes, self.edges)

        bfs_agent = BFS(graph, self.bfs_node)
        bfs_agent.search()
        bfs_agent.visualize()

    def MST_status(self) :

        # Visualize the MST
        self.mst_solver.visualize_mst(self.nodes)

        # Get nodes in the MST
        mst_nodes = self.mst_solver.get_mst_nodes()

        # Print MST details
        print("Minimum Spanning Tree:")
        total_weight = self.mst_solver.get_mst_total_weight()

        for edge in self.mst_solver.mst_edges:
            print(f"Edge: {edge.source.id} - {edge.target.id}, Weight: {edge.weight}")

        print(f"Total MST Weight: {total_weight}")

        # Optional: Visualize or further process the MST
        self.mst_solver.mst_nodes, self.mst_solver.mst_edges = mst_nodes, self.mst_solver.mst_edges

    def PerformMst(self) :
    # Create MST solver
        # Compute Minimum Spanning Tree
        self.mst_solver.compute_mst(self.nodes, self.edges)
        self.MST_status()

    def AddNodeToMST(self, node, edges) :
        self.mst_solver.add_node(node, edges)
        self.MST_status()

    def add_node(self):
        x, y = np.random.rand(2) * 10
        new_pos = np.array([[x, y]])
        self.pos = np.vstack([self.pos, new_pos]) if self.pos.size else new_pos
        node_id = len(self.nodes)
        new_node = Node(node_id, (x, y), np.random.randint(4))
        # Add new edges
        New_edges = []
        edges_count = np.random.randint(1,5)
        for _ in range(edges_count) :
            edge_weight = np.random.randint(1, 20)
            edge_cap = np.random.randint(5, 30)

            New_edges.append(Edge(random.choice(self.nodes), new_node, edge_weight, edge_cap))

        self.edges.extend(New_edges)
        self.nodes.append(new_node)
        self.update_graph()

        self.AddNodeToMST(new_node, New_edges)

    def update_graph(self):
        if self.edges:
            adj = np.array([[e.source.id, e.target.id] for e in self.edges])
        else:
            adj = None

        symbols = ['o'] * len(self.nodes)
        sizes = [10] * len(self.nodes)
        brushes = [ZONE_COLORS.get(node.zone, 'w') for node in self.nodes]

        # Color logic
        if self.show_traffic:
            pens = []
            for edge in self.edges:
                edge.set_traffic_color()
                pens.append(edge.color)
                # print(f" {edge.color} for edge {edge.source.id} <-> {edge.target.id}")
        else:
            pens = [pg.mkPen('k').color().getRgb()] * len(self.edges)

        pens = np.array(pens, dtype=np.uint8)

        self.graph_item.setData(
            pos=self.pos,
            adj=adj,
            size=sizes,
            symbol=symbols,
            pxMode=True,
            pen=pens,
            brush=brushes
        )



    def find_border_nodes(self):
        # Get unique zones from the nodes
        zones = sorted(set(node.zone for node in self.nodes if node.zone is not None))

        # Generate all zone-to-zone keys
        zone_pairs = [f"{a}to{b}" for a, b in combinations(zones, 2)]
        border_dict = {key: None for key in zone_pairs}

        # Temporary storage with sets to avoid duplicates
        temp_dict = defaultdict(set)

        for edge in self.edges:
            n1, n2 = edge.nodes
            z1, z2 = n1.zone, n2.zone

            if z1 != z2:
                key = f"{min(z1, z2)}to{max(z1, z2)}"
                temp_dict[key].add(n1)
                temp_dict[key].add(n2)

        # Update final dict with actual values where border nodes exist
        for key in temp_dict:
            border_dict[key] = list(temp_dict[key])

        self.border_nodes = border_dict

    def generate_map(self):

        total_nodes = 100
        self.nodes = [Node(i, (0, 0),) for i in range(total_nodes)]  # positions will be updated

        tree_edges = self.generate_random_spanning_tree(total_nodes)

        all_edges = set(tree_edges)
        G = nx.Graph()
        G.add_edges_from(all_edges)
        pos_dict = nx.spring_layout(G, seed=42, scale=100)
        positions = np.array([pos_dict[i] for i in range(total_nodes)])
        max_distance = np.linalg.norm(positions.max(axis=0) - positions.min(axis=0))

        extra_edges = int(total_nodes * 2.5)
        attempts = 0
        max_attempts = 5000

        while len(all_edges) < len(tree_edges) + extra_edges and attempts < max_attempts:
            a, b = np.random.choice(total_nodes, 2, replace=False)
            if a == b or (a, b) in all_edges or (b, a) in all_edges:
                attempts += 1
                continue

            dist = np.linalg.norm(positions[a] - positions[b])
            prob = np.exp(-dist / (max_distance * 0.25))

            if np.random.rand() < prob:
                all_edges.add((a, b))
                G.add_edge(a, b)
            attempts += 1

        pos_dict = nx.spring_layout(G, seed=42, scale=100,)
        for k in pos_dict:
            pos_dict[k] = np.round(pos_dict[k] / 10) * 10

        self.pos = np.array([pos_dict[i] for i in range(total_nodes)])
        center = self.pos.mean(axis=0)
        self.pos -= center

        self.assign_zone()

        self.edges = [Edge(self.nodes[a], self.nodes[b], np.random.randint(1, 20),np.random.randint(5, 30)) for (a, b) in all_edges]
        self.find_border_nodes()
        print(self.border_nodes)
        self.PerformMst()
        self.update_graph()

        self.focus_on_zone(0)  # Focus on North

        # Start the clock after 3.5 seconds if not already started
        if not self.clock_started:
            QTimer.singleShot(3500, self.start_clock_after_map)
            self.clock_started = True

    def start_clock_after_map(self):
        self.clock.start()

    def assign_zone(self):
        for i, node in enumerate(self.nodes):
            node.pos = self.pos[i]

        for i, node in enumerate(self.nodes):
            node.pos = self.pos[i]
            x, y = node.pos

            if abs(y) > abs(x):  # prioritize N/S over E/W
                if y >= 0:
                    node.zone = 0  # North
                else:
                    node.zone = 1  # South
            else:
                if x >= 0:
                    node.zone = 2  # East
                else:
                    node.zone = 3  # West

    def focus_on_zone(self, zone_id):
        zone_nodes = [n for n in self.nodes if n.zone == zone_id]
        if not zone_nodes:
            return
        coords = np.array([n.pos for n in zone_nodes])
        min_xy = coords.min(axis=0)
        max_xy = coords.max(axis=0)
        pad = 10
        self.view.setRange(
            xRange=(min_xy[0] - pad, max_xy[0] + pad),
            yRange=(min_xy[1] - pad, max_xy[1] + pad)
        )

    def generate_random_spanning_tree(self, n):
        edges = []
        unvisited = set(range(n))
        visited = {unvisited.pop()}

        while unvisited:
            u = np.random.choice(list(visited))
            v = np.random.choice(list(unvisited))
            edges.append((u, v))
            visited.add(v)
            unvisited.remove(v)

        return edges

    def on_click(self, event):
        if self.pos.size == 0:
            return

        mouse_point = self.view.mapSceneToView(event.scenePos())
        click_x, click_y = mouse_point.x(), mouse_point.y()

        distances = np.linalg.norm(self.pos - np.array([click_x, click_y]), axis=1)
        closest_index = np.argmin(distances)

        pens = []
        if self.edges:
            adj = np.array([[e.source.id, e.target.id] for e in self.edges])
        else:
            adj = None
        symbols = ['o'] * len(self.nodes)
        sizes = [10] * len(self.nodes)

        brushes = [ZONE_COLORS.get(node.zone, 'w') for node in self.nodes]

        if distances[closest_index] > 3:  # Ignore if too far from a node
            for _ in self.edges:
                pens.append((0, 0, 0, 255))  #black
            self.graph_item.setData(
                pos=self.pos,
                adj=adj,
                size=sizes,
                symbol=symbols,
                pxMode=True,
                pen=np.array(pens),
                brush=brushes
            )
            self.assign_zone()
            return

        print(f"Clicked node: {closest_index}")
        if self.travel_mode:
            if not self.travel_path or self.travel_path[-1] != closest_index:
                self.travel_path.append(closest_index)

        if self.bfs_mode :
            self.bfs_node = self.nodes[closest_index]

        self.highlight_edges(closest_index)

    def highlight_edges(self, node_index):
        if self.edges:
            adj = np.array([[e.source.id, e.target.id] for e in self.edges])
        else:
            adj = None
        symbols = ['o'] * len(self.nodes)
        sizes = [10] * len(self.nodes)

        brushes = [ZONE_COLORS.get(node.zone, 'w') for node in self.nodes]

        pens = []
        for edge in self.edges:
            node_color = ZONE_COLORS.get(edge.source.zone, 'w')
            edge_color = COLORS.get(node_color, (0, 0, 0, 255))
            if node_index == edge.source.id or node_index == edge.target.id:
                pens.append((0, 0, 0, 255))
            else:
                pens.append((0, 0, 0, 0))  # nothing

        pens = np.array(pens, dtype=np.uint8)  # <-- convert to numpy array

        self.graph_item.setData(
            pos=self.pos,
            adj=adj,
            size=sizes,
            symbol=symbols,
            pxMode=True,
            pen=pens,
            brush=brushes
        )


if __name__ == '__main__':
    app = QApplication(sys.argv)

    clock = ClockGenerator()

    win = GraphApp(clock)
    win.show()
    sys.exit(app.exec())
    win.show()
    sys.exit(app.exec())
