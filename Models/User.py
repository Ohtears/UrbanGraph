from this import s
from Models.data_obj import Node, Edge

class User:

    id_incr = 0

    def __init__(self, current_loc=None):
        self.current_loc = current_loc
        self.travel_nodes = []
        self.travel_path = []
        self.node_index = 0
        self.path_index = 0
        self.done = True
        self.id = User.id_incr
        User.id_incr += 1

    def set_route(self, nodes, path):
        self.travel_nodes = nodes
        self.travel_path = path
        self.node_index = 0
        self.path_index = 0
        self.done = False

    def tick(self):
        if self.done:
            return

        # Clean up from previous edge
        if isinstance(self.current_loc, Edge):
            self.current_loc.passengers.discard(self)

        # Step along edge if needed
        if self.path_index < self.node_index and self.path_index < len(self.travel_path):
            edge = self.travel_path[self.path_index]
            edge.passengers.add(self)
            self.current_loc = edge
            self.path_index += 1

        # Step to next node
        elif self.node_index < len(self.travel_nodes):
            self.current_loc = self.travel_nodes[self.node_index]
            self.node_index += 1

        # Done traveling
        else:
            self.done = True

    def is_active(self):
        return not self.done

    def getLoc(self):
        if isinstance(self.current_loc, Node):
            return f"User{self.id} is in Node {self.current_loc}"
        elif isinstance(self.current_loc, Edge):
            return f"User{self.id} is on Edge {self.current_loc}"
        return "User location unknown"
        