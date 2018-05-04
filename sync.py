#!/usr/bin/env python
# encoding: utf-8

#sync all files to destine

'''
please set ssh config:

~/.ssh/config 
host *
   KexAlgorithms +diffie-hellman-group1-sha1
   HOSTKeyAlgorithms=+ssh-dss
ControlMaster auto
ControlPath ~/.ssh/master-%r@%h:%p
then:
chmod 600 ~/.ssh/config 
'''

import os
import sys


def get_files(rootDir, suffix, lastupdate=0):
    result = []
    #pwd = os.getcwd()
    for root, dirs, files in os.walk(rootDir):
         result += [root+"/"+f for f in files if f.split('.')[-1] in suffix]
         
    result = filter(lambda x : os.stat('.').st_mtime > lastupdate, result)
    
    return result



def sync(files, dest_dir, dest_host, dest_port='36000', dest_user='root'):
     
     for f in files:
        dest_path = dest_dir + f
        cmd = "scp -P {port} '{files}' '{user}@{ip}:{path}'" \
                  .format(port=dest_port, files=f, user=dest_user, ip=dest_host, path=dest_path)
        ret = os.system(cmd)
        #print("cmd:%s, ret:%d" %(cmd, ret))
        if ret != 0:
           print("copy failed")
           sys.exit(-1)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('usage: %s ip [port] [destdir]' % sys.argv[0])
    
    files = get_files('.', ['py'], 0)
    if len(files) == 0:
        print("No files to sync")
    dest_host=sys.argv[1]
    dest_dir='/root/' +  os.getcwd().split('/')[-1] + "/"
    sync(files, dest_dir, dest_host)

    
