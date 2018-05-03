#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


from sanic import Sanic
from sanic.response import json as sanic_response


import ujson
import requests
from inspect import signature
import logging
import datetime

FORMAT = '%(asctime)-15s %(levelname)s %(process)d %(message)s %(filename)s:%(module)s:%(funcName)s:%(lineno)d'
handlers=[logging.StreamHandler()]
logging.basicConfig(format=FORMAT, level=logging.DEBUG, handlers = handlers)
    
'''
Rpc Server Warpper, every for easy used.
1. add Rpc Method:
   @Rpc.method() 
   def test(a, b, c):
       print ("a:%s b:%s c:%s" % (a,b,c))
       return {"hello": "world"}
2. start server:
   srv = Rpc.newServer(8000)    
   srv.run()
    
3. client call
   client = Rpc.newClient('127.0.0.1', 8000)
   print(client.test(1, 2, 3))
    
'''


class RpcClient:
    def __init__(self, ip, port, rpc_signature):
        self.ip = ip
        self.port = port
        for name,signature in rpc_signature.items():
            self.__setattr__(name, self._client_wrapper(name, signature))
    
    def _client_wrapper(self, rpc_name, rpc_signature):
        def _rpc_call(*args, **kargs):
            for k,v in zip(rpc_signature.parameters, args):
                 kargs[k] = v
            res = requests.post("http://%s:%s/%s" % (self.ip, self.port, rpc_name), params=kargs)
            return res.json() if res.status_code == 200 else None
        return _rpc_call


class RpcServer:
     def __init__(self, port, methods):
         self.port = port 
         self.app  = Sanic()
         for k,v in methods.items():
             self.app.add_route(v, "/" + k,  methods=['GET', 'POST'])
     def run(self):
         self.app.run(host="0.0.0.0", port=8000)

class Rpc:
    rpc_handlers  = {}
    rpc_signature = {}   
    
    @classmethod
    def newClient(cls, ip, port):
        return RpcClient(ip, port, Rpc.rpc_signature)
    @classmethod
    def newServer(cls, port):
        return RpcServer(port, Rpc.rpc_handlers)
      
    @staticmethod
    def _server_wrapper(handler):
        async def unpack_params_handler(request):
              now  = datetime.datetime.now()
              args =  {k : request.args.get(k) for k in request.args} if len(request.body) == 0 else request.json
              res  = handler(**args)
              logging.info("%s %s(%s) -> %s start: %s, cost: %s" % \
                             (request.ip, request.url, args, res, \
                              now.strftime("%Y%m%d %H:%M:%S.") + "%03d"%(now.microsecond/1000), datetime.datetime.now()-now))
              return sanic_response(res)
              
        return unpack_params_handler
        
    @classmethod
    def method(cls):
        def request_unpack_wrapper(handler):
            cls.rpc_handlers[handler.__name__]  = Rpc._server_wrapper(handler)
            cls.rpc_signature[handler.__name__] = signature(handler)
            logging.info("Rpc register %s%s" %(handler.__name__, signature(handler)))
            return handler
        return request_unpack_wrapper
    
''' Rpc Method implement, must be static method '''
@Rpc.method()
def test(a, b, c):
   print ("a:%s b:%s c:%s" % (a,b,c))
   return {"hello": "world"}


@Rpc.method()
def getStatus():
   return {"status": 0}

if __name__ == "__main__":
    srv = Rpc.newServer(8000)
    srv.run()

