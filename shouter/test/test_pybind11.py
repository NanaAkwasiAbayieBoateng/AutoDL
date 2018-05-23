import sys
sys.path.append('build/lib.linux-x86_64-3.5')

import example

print(dir(example))

print(example.add(1, 2))


p = example.Pet('Molly')
print(p.name)

p.setVec([1,2,3])
p.name = 'Charly'
print(p.name)

p = example.Dog('Molly')
print(p.name)


p.name = 'Charly'
print(p.name)
