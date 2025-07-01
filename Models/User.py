from this import s
from Models.data_obj import Node, Edge
import time
class User :
    def __init__(self, current_loc=None) :
        self.current_loc = current_loc

    def getLoc(self) :

        if isinstance(self.current_loc, Node) :
            return f"User is in the Node {self.current_loc}"
        if isinstance(self.current_loc, Edge) :
            return f"User is in the path {self.current_loc}"

    def travel(self, nodes, path, clock, log) :
        i = 0
        j = 0
        while i < len(path) or j < len(nodes) :
            if clock.get_clock_value() == 1 :
                if i < j :
                    self.current_loc = path[i]
                    path[i].passengers.add(self)
                    i += 1
                else :
                    if isinstance(self.current_loc, Edge):
                        self.current_loc.passengers.discard(self)
                    self.current_loc = nodes[j]
                    j += 1
                print(self.getLoc()) if log else None
                # print("Edges passengers count :")
                # for edge in path :
                #     print(len(edge.passengers), end=" ")
                # print()
                time.sleep(1)
