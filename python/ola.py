>>> def intcaststr(bitlist):
...     return int("".join(str(i) for i in bitlist), 2)
... 
>>> def intcastlookup(bitlist):
...     return int(''.join('01'[i] for i in bitlist), 2)
... 
>>> def shifting(bitlist):
...     out = 0
...     for bit in bitlist:
...         out = (out << 1) | bit
...     return out
... 
>>> timeit.timeit('convert([1,0,0,0,0,0,0,0])', 'from __main__ import intcaststr as convert', number=100000)
0.5659139156341553
>>> timeit.timeit('convert([1,0,0,0,0,0,0,0])', 'from __main__ import intcastlookup as convert', number=100000)
0.4642159938812256
>>> timeit.timeit('convert([1,0,0,0,0,0,0,0])', 'from __main__ import shifting as convert', number=100000)
0.1406559944152832
