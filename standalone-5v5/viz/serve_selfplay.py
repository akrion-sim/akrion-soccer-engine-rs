#!/usr/bin/env python3
"""Serve the self-play champion viewer on :8081. /newgame records a fresh
champion-vs-champion match with the current best policy."""
import os, subprocess, tempfile
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
OUT=os.path.join(ROOT,"out"); BIN=os.path.join(ROOT,"target","release","fiveaside")
def champ():
    if os.path.exists(os.path.join(OUT,"actor.txt")): return OUT
    cdir=os.path.join(OUT,"champions"); gens=[]
    if os.path.isdir(cdir):
        for d in os.listdir(cdir):
            if d.startswith("gen") and d[3:].isdigit(): gens.append((int(d[3:]),os.path.join(cdir,d)))
    gens.sort(); return gens[-1][1] if gens else OUT
class H(SimpleHTTPRequestHandler):
    def __init__(self,*a,**k): super().__init__(*a,directory=OUT,**k)
    def log_message(self,*a): pass
    def do_GET(self):
        u=urlparse(self.path)
        if u.path=="/newgame":
            seed=parse_qs(u.query).get("seed",["7"])[0]
            fd,tmp=tempfile.mkstemp(suffix=".json"); os.close(fd); c=champ()
            r=subprocess.run([BIN,"play",str(int(seed)%2**63),"--out-dir",c,"--opponent",c,"--out",tmp],
                             capture_output=True,text=True,timeout=120)
            body=open(tmp,"rb").read() if r.returncode==0 else b"{}"
            os.unlink(tmp)
            self.send_response(200 if r.returncode==0 else 500)
            self.send_header("content-type","application/json"); self.send_header("content-length",str(len(body))); self.end_headers()
            self.wfile.write(body); return
        return super().do_GET()
PORT=int(os.environ.get("PORT","8081"))
print(f"self-play champion viewer: http://127.0.0.1:{PORT}/")
ThreadingHTTPServer(("127.0.0.1",PORT),H).serve_forever()
