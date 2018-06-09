import logging
import datetime
import sys

from logging.handlers import TimedRotatingFileHandler


"""
As keep simple and keep scalability, just wrapper the min logic
this just a configure file
"""
#class TraceHandker(TimedRotatingFileHandler):
#TraceHandker(filename='a.log', when='h', interval=1)
#class TraceHandker(logging.StreamHandler):
class StreamTraceHandker(logging.StreamHandler):
    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)

    def emit(self, record):
        # 7 is get by debug
        frame = sys._getframe(7)
        code = frame.f_code
        record.msg = record.msg + " " + ":".join((code.co_name, code.co_filename,str(frame.f_lineno),))
        super().emit(record)

class FileTraceHandker(logging.StreamHandler):
    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)

    def emit(self, record):
        # 7 is get by debug
        frame = sys._getframe(7)
        code = frame.f_code
        record.msg = record.msg + " " + ":".join((code.co_name, code.co_filename,str(frame.f_lineno),))
        super().emit(record)
        
#FORMAT = '%(asctime)-15s %(name)s %(process)d %(levelname)s %(message)s'
FORMAT = '%(asctime)-15s %(process)d %(levelname)s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)
handlers=[StreamTraceHandker()]

logging.basicConfig(format=FORMAT, level=logging.DEBUG, handlers = handlers)


def getLogger(name, path):
    log_filename = path + "/" + name + "_" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.log'
    handler = FileTraceHandker(filename=log_filename)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)-15s %(name)s %(process)d %(levelname)s %(message)s')
    handler.setFormatter(formatter)  

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    return logger

    
