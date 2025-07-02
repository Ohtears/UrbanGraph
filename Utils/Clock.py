import threading
import time

class ClockGenerator:
    def __init__(self, off_time = 2, on_time =1):
        self.running = False
        self.clock_value = 0
        self.off_time = off_time
        self.on_time = on_time
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._generate_clock)
        self.thread.setDaemon(True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def _generate_clock(self):
        while self.running:
            # 0 for 5 seconds
            self.clock_value = 0
            # print(f"Clock value: {self.clock_value}")
            time.sleep(self.off_time)

            # 1 for 1 second
            self.clock_value = 1
            # print(f"Clock value: {self.clock_value}")
            time.sleep(self.on_time)

    def get_clock_value(self):
        return self.clock_value

