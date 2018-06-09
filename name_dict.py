#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# please use the python3.5+

import yaml
import datetime
import threading
import os
import subprocess
import shlex
import hashlib

'''
import this first.

'''

class NameDict(dict):
    """easy dict key access for d['k'] -> d.k"""
    def __init__(self, *args, **kwargs):
        """for NameDict({aa:bb}, aa=c, bb=x)"""
        for p in args:
            for k in p:
                self.__setitem__(k, p[k])
        for k in kwargs:
            self.__setitem__(k, kwargs[k])

    def __getattr__(self, name):
        if name.startswith('__'):
            return super().__getattr__(name)
        if name.startswith('has_'):
            return lambda: name[4:] in self
        if name == 'ensure':
            return self.ensure
        if name in self:
            return self[name]
        
        # set a empty
        self.__setitem__(name, '')
        return ''


    def __setitem__(self, key, value):
        """for d[k] == [a,b,c]"""
        if key.startswith('__'):
            return super().__setitem__(key, value)
        if isinstance(value, dict):
            return super().__setitem__(key, NameDict(**value))
        if isinstance(value, list):
            val = [NameDict(**k) if isinstance(k, dict) else k for k in value]
            return super().__setitem__(key, val)
        if isinstance(value, tuple):
            val = tuple((NameDict(**k) if isinstance(k, dict) else k for k in value))
            return super().__setitem__(key, val)
        return super().__setitem__(key, value)

    def __setattr__(self, name, value):
        """for d.s = a, b, c"""
        if name.startswith('__'):
            return super().__setattr__(name, value)
        return self.__setitem__(name, value)

    def __str__(self):
        return str(super())

    def ensure(self, name, default, valid=None):
        if name not in self:
            self.__setattr__(name, default)
        if valid is not None and self.__getattr__(name) not in valid:
            self.__setattr__(name, default)
        return self[name]

'''
this enable when yaml add:
--- !NameDict

otherwise the default is:
!!python/object/new:name_dict.NameDict
'''
def NameDict_constructor(loader, node):
    values = loader.construct_mapping(node, deep=True)
    #print(type(values))
    return NameDict(values)
    
yaml.add_constructor(u'!NameDict', NameDict_constructor)

class Helper:
    """some trivial functions"""
    @staticmethod
    def function_wrapper(args):
        """A wrapper to class method as pool_map_wrapper(Worker.do, w, cmd ..)"""
        fun, c, *a = args
        return fun(c, *a)

    @staticmethod
    def current_timestamp():
        now = datetime.datetime.now()
        return now.strftime("%Y-%m-%d-%H-%M-%S")

    @staticmethod
    def inspect(o):
        print("type:%s, repr:%s" % (type(o), o.__repr__()))
        print("doc:%s" % o.__doc__)
        inners = [k for k in dir(o) if k.startswith('_')]
        members = [k for k in dir(o) if not k.startswith('_')]
        print("inner: %s" % ','.join(inners))
        for m in members:
            try:
                t = o.__getattribute__(m)
                if type(t) in set([type(i) for i in [{}, tuple(), [], "", 1, 1.0]]):
                    print("%s:%s" % (m, t))
                elif type(t) == type(open):
                    print("%s:%s" % (m, t))
                elif isinstance(t, object):
                    print("%s:%s:%s" % (m, type(t), t))
                else:
                    print("%s:%s" % (m, type(t)))
            except Exception as e:
                print(e)

    @staticmethod
    def get_zgkm_home():
        return os.environ['ZGKM_HOME']

    @staticmethod
    def merge_dir(src, dst):
        cmd = "cp -r %s/* %s" % (src, dst)
        p = subprocess.Popen(cmd, shell=True)
        p.wait()

    @staticmethod
    def split_ext(path):
        for ext in ['.tar.gz', '.tar.bz2']:
            if path.endswith(ext):
                return path[:-len(ext)], path[-len(ext):]
        return os.path.splitext(path)

    @staticmethod
    def md5(data):
        if isinstance(data, str):
            data = data.encode('utf-8')
        hash_md5 = hashlib.md5(data)
        return hash_md5.hexdigest()


class YtGpuStat(object):
    def __init__(self):
        super(YtGpuStat, self).__init__()

    @staticmethod
    def is_idle():
        # cmd = "nvidia-smi --query-compute-apps=name  --format=csv,noheader"
        cmd = "ls"
        p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        out = out.decode('utf-8')
        return out.find('python') < 0


class YtPeriodTask:
    """for timer task"""
    def __init__(self, interval, func, *args):
        self.interval = interval
        self.func = func
        self.args = args
        self.timer = None

    def start(self):
        self.timer = threading.Timer(self.interval, self.exec)
        self.timer.start()

    def exec(self):
        self.func(*self.args)
        self.timer = threading.Timer(self.interval, self.exec)
        self.timer.start()

    def cancel(self):
        self.timer.cancel()
        self.timer = None


class YtPeriodTaskEvent:
    def __init__(self, interval, func, *args):
        self.interval = interval
        self.func = func
        self.args = args
        self.event = None
        self.thread = None

    def start(self):
        self.event = threading.Event()
        self.thread = threading.Thread(target=self.loop)
        self.thread.start()

    def loop(self):
        self.func(*self.args)
        while not self.event.wait(self.interval):
            self.func(*self.args)

    def cancel(self):
        if self.event is not None:
            self.event.set()
        if self.thread is not None:
            self.thread.join()

if __name__ == '__main__':
    
    with open('benchmarks/cifar10_train.yaml') as f:
       conf = yaml.load(f)
    #print("conf.name:%s, conf.data:%s" % (conf.name, conf.data))
    print(type(conf))
    print(yaml.dump(conf, default_flow_style=False))
