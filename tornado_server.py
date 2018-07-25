# coding=utf-8
from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from web_server import app
import webbrowser
webbrowser.open("http://127.0.0.1:5000")
http_server = HTTPServer(WSGIContainer(app))
http_server.listen(5000)  # flask默认的端口
IOLoop.instance().start()