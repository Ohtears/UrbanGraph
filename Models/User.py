class User :
    def __init__(self) :
        self.current_loc = None
        self.current_edge = None

    def travel(self, dsts) :
        if len(dsts) < 1 :
            return None
        if len(dsts) == 1 :
            pass
