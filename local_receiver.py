"""Background localhost receiver; keep newest poses and every button edge."""
import collections
import json
import socket
import threading

def newest_pose_with_button_edges(messages):
    """Coalesce stable-button runs; never remove press/release or tracking transitions."""
    selected=[]
    last_key=None
    for m in messages:
        buttons=m.get("buttons",{})
        key=(m.get("session"),tuple(sorted(buttons.items())),m.get("status"),
             m.get("ar_quality"),m.get("position_m") is None,m.get("quaternion_wxyz") is None)
        if selected and key==last_key:
            selected[-1]=m
        else:
            selected.append(m)
        last_key=key
    return selected

class LocalReceiver:
    def __init__(self, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1",port))
        self.sock.settimeout(.1)
        self.queue = collections.deque()
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self.read, daemon=True)
        self.thread.start()
    def read(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(32768)
                m = json.loads(data)
                if not isinstance(m,dict):
                    continue
                with self.lock:
                    if len(self.queue) >= 512:
                        self.queue.clear()
                        self.queue.append({})
                    self.queue.append(m)
            except socket.timeout:
                pass
            except (ValueError, OSError):
                pass
    def drain(self):
        with self.lock:
            items = list(self.queue)
            self.queue.clear()
            return newest_pose_with_button_edges(items)
    def close(self):
        self.running = False
        self.thread.join(timeout=.3)
        self.sock.close()
