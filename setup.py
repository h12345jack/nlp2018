#coding=utf8
import sys
from distutils.core import setup
import sys
from cx_Freeze import setup, Executable
import tornado
import flask

build_exe_options = {"packages": ["numpy", "tornado", "flask", "asyncio", "tqdm",
                                  "jinja2","flask_cors", 
                                    ], "excludes": ["tkinter", "matplotlib", "pandas"]}

base = None
# if sys.platform == "win32":
#     base = "Win32GUI"

setup(  name = "nlp2018",
        version = "0.1",
        description = "nlp2018",
        options = {"build_exe": build_exe_options},
        executables = [Executable("web_server.py", base=base)])