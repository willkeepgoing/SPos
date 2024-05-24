from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import os
from multiprocessing import Process
from socketserver import ThreadingMixIn


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    pass


class Resquest(BaseHTTPRequestHandler):
    def do_GET(self):
        url_path = self.path[1:]
        try:
            with open(url_path, 'rb') as page_file:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(page_file.read())
        except:
            page_file = open('examples/404.html', 'rb')
            self.send_response(200)
            self.end_headers()
            self.wfile.write(page_file.read())

    def do_POST(self):
        # 处理上传的视频
        start_time = self.path.split('/')[-1]
        client_name = start_time
        req_datas = self.rfile.read(int(self.headers['content-length']))
        self.send_response(200)
        # self.send_header('Content-type', 'application/json')
        self.end_headers()
        input_file = client_name
        output_file = client_name + '.mp4'
        with open(input_file, 'wb') as file:
            file.write(req_datas)
        fin = open(input_file, 'rb')
        a = fin.readlines()
        fout = open(input_file, 'wb')
        b = a[4:-1]
        fout.writelines(b)
        os.rename(input_file, output_file)

        os.system('python frame.py --func video --frame_gap 5 --video_path /{}'.format(output_file))
        os.remove(output_file)
        res_data = "666"


def func(output_file):
    os.system('python frame.py --func video --frame_gap 5 --video_path /{}'.format(output_file))
    os.remove(output_file)


# 开启服务器，可处理接受视频的get/post请求
host = ('59.72.63.157', 8990)
server = ThreadedHTTPServer(host, Resquest)
server.allow_reuse_address = True
print("服务端已开启, 监听   %s:%s" % host)
server.serve_forever()
