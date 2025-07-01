import sys
import os
import numpy as np
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QGraphicsPixmapItem
import pyqtgraph as pg
import networkx as nx

# Add project root to sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Models.data_obj import Node, Edge, Graph, ZONE_COLORS, COLORS
from Utils.MST import PrimMST
from Utils.AStar import AStar

class GraphApp(QWidget):
    def __init__(self):
        super().__init__()

        bg_pixmap = QPixmap("BackgroundImg.png")

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
        self.pos = np.array([])

        self.mst_solver = PrimMST()

        layout = QVBoxLayout()
        layout.addWidget(self.graph_widget)

        self.travel_mode = False
        self.travel_path = []

        self.travel_btn = QPushButton("Travel")
        self.travel_btn.clicked.connect(self.start_travel)
        layout.addWidget(self.travel_btn)

        self.done_btn = QPushButton("Done")
        self.done_btn.clicked.connect(self.finish_travel)
        self.done_btn.hide()
        layout.addWidget(self.done_btn)

        self.add_node_btn = QPushButton("Add Node")
        self.add_node_btn.clicked.connect(self.add_node)
        layout.addWidget(self.add_node_btn)

        self.generate_btn = QPushButton("Generate Random Map")
        self.generate_btn.clicked.connect(self.generate_map)
        layout.addWidget(self.generate_btn)

        self.setLayout(layout)

    def start_travel(self):
        self.travel_mode = True
        self.travel_path = []
        self.travel_btn.hide()
        self.done_btn.show()
        print("Travel mode activated. Click on nodes to record your path.")

    def finish_travel(self):
        self.travel_mode = False
        self.done_btn.hide()
        self.travel_btn.show()
        print("Travel path completed:")
        print(" -> ".join(str(node_id) for node_id in self.travel_path))
        astar = AStar(self.nodes, self.edges)
        print(self.nodes[self.travel_path[0]].id, self.nodes[self.travel_path[1]].id)
        nodes, path = astar.a_star_search(self.nodes[self.travel_path[0]],self.nodes[self.travel_path[1]])
        print(nodes)
        print(path)

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
            edge_weght = np.random.randint(1, 20)
            edge_cap = np.random.randint(5, 30)

            New_edges.append(Edge(random.choice(self.nodes), new_node, edge_weght, edge_cap))


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

        # Use zone to set brush color
        brushes = [ZONE_COLORS.get(node.zone, 'w') for node in self.nodes]  # fallback to white

        self.graph_item.setData(
            pos=self.pos,
            adj=adj,
            size=sizes,
            symbol=symbols,
            pxMode=True,
            pen=pg.mkPen('k'),
            brush=brushes
        )


    def generate_map(self):

        total_nodes = 20
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

        self.PerformMst()
        self.update_graph()

        self.focus_on_zone(0)  # Focus on North


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
            for edge in self.edges:
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
                pens.append(edge_color)
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
    win = GraphApp()
    win.show()
    sys.exit(app.exec())
    win.show()
    sys.exit(app.exec())
