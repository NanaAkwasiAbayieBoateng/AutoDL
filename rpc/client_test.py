#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import yaml
import threading
import datetime
import sys
import logging
import time

from ARpcServer import Rpc

if __name__=='__main__':
    client = Rpc.newClient('127.0.0.1', 8000)
    print(client.test(1, 2, 3))
    print(client.getStatus())

