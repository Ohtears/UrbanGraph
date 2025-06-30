import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QGraphicsPixmapItem
import pyqtgraph as pg
import networkx as nx

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

        # Create image item and scale it (optional, depending on layout)
        bg_item = QGraphicsPixmapItem(bg_pixmap)
        bg_item.setZValue(-10)  # ensures background is behind graph

        # Optionally transform the image to match your scene scale
        # self.bg_rect = bg_pixmap.rect()
        bg_rect = bg_pixmap.rect()
        bg_item.setTransformOriginPoint(bg_rect.width() / 2, bg_rect.height() / 2)
        bg_item.setScale(0.2)  # start with scale=1

        # Shift to center
        bg_item.setPos(-bg_rect.width() / 2, -bg_rect.height() / 2)

        # Add background to view
        self.view.addItem(bg_item)

        self.graph_item = pg.GraphItem()
        self.view.addItem(self.graph_item)

        self.nodes = []
        self.edges = []
        self.pos = np.array([])

        layout = QVBoxLayout()
        layout.addWidget(self.graph_widget)

        self.add_node_btn = QPushButton("Add Node")
        self.add_node_btn.clicked.connect(self.add_node)
        layout.addWidget(self.add_node_btn)

        self.generate_btn = QPushButton("Generate Random Map")
        self.generate_btn.clicked.connect(self.generate_map)
        layout.addWidget(self.generate_btn)

        self.setLayout(layout)

    def add_node(self):
        x, y = np.random.rand(2) * 10
        new_pos = np.array([[x, y]])
        self.pos = np.vstack([self.pos, new_pos]) if self.pos.size else new_pos
        self.nodes.append(len(self.nodes))
        if len(self.nodes) > 1:
            self.edges.append((len(self.nodes) - 2, len(self.nodes) - 1))
        self.update_graph()

    def update_graph(self):
        adj = np.array(self.edges) if self.edges else None
        symbols = ['o'] * len(self.nodes)
        sizes = [10] * len(self.nodes)
        self.graph_item.setData(
            pos=self.pos, adj=adj, size=sizes, symbol=symbols,
            pxMode=True, pen=pg.mkPen('k'), brush='c'
        )
    def generate_map(self):
        total_nodes = 100
        self.nodes = list(range(total_nodes))

        # Step 1: Generate a full spanning tree manually
        tree_edges = self.generate_random_spanning_tree(total_nodes)

        # Step 2: Create graph and apply initial layout
        all_edges = set(tree_edges)
        G = nx.Graph()
        G.add_edges_from(all_edges)
        pos_dict = nx.spring_layout(G, seed=42, scale=100)
        positions = np.array([pos_dict[i] for i in range(total_nodes)])
        max_distance = np.linalg.norm(positions.max(axis=0) - positions.min(axis=0))

        # Step 3: Add extra edges biased toward shorter distance
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

        # Final layout with updated edges
        # Final layout with full graph
        pos_dict = nx.spring_layout(G, seed=42, scale=100,)  # try increasing k
        # Optional: snap positions to a grid (for urban feel)
        for k in pos_dict:
            pos_dict[k] = np.round(pos_dict[k] / 10) * 10

        self.pos = np.array([pos_dict[i] for i in range(total_nodes)])
        # Center the layout around (0, 0)
        center = self.pos.mean(axis=0)
        self.pos -= center

        self.edges = list(all_edges)

        self.update_graph()

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
        adj = np.array(self.edges) if self.edges else None
        symbols = ['o'] * len(self.nodes)
        sizes = [10] * len(self.nodes)


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
            brush='c'
            )
            return

        print(f"Clicked node: {closest_index}")
        self.highlight_edges(closest_index)

    def highlight_edges(self, node_index):
        adj = np.array(self.edges) if self.edges else None
        symbols = ['o'] * len(self.nodes)
        sizes = [10] * len(self.nodes)

        pens = []
        for edge in self.edges:
            if node_index in edge:
                pens.append((0, 255, 0, 255))  # green
            else:
                pens.append((0, 0, 0, 0))  #nothing

        self.graph_item.setData(
            pos=self.pos,
            adj=adj,
            size=sizes,
            symbol=symbols,
            pxMode=True,
            pen=np.array(pens),
            brush='c'
        )
        


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = GraphApp()
    win.show()
    sys.exit(app.exec())
