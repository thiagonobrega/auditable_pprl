'''
Created on 15 de jun de 2016

@author: Thiago Nobrega
@mail: thiagonobrega@gmail.com
'''
import ngram
import threading
import multiprocessing
import hashlib

class NullField():
    """
        Classe utilizada quando o valor  é nulo
    """
    def __init__(self,nome):
        self.nome = nome
        self.sim = 0

class ColumnEncrypter(threading.Thread):
    """
        Class to Encrypt data [list of list]
        
        rember to set data (set_rows) before start to encrypt the data 
    """
    
    def __init__(self, column, cryptofun , bigrams = False, md5 = False, lock = threading.Lock(), **cryptfun_args):
            """
                Encrypt row
                row : a list of data to be encrupt
                columns : a list of collumn in row to encrypt
                cryptofun : Crypto function
                bigrams : transform data in bigrams? Defaul False
                md5 : encrypt data using md5
                **cryptfun_args : arguments for crypto function
            """
            threading.Thread.__init__(self)
            self.thread = threading.Thread(target=self.run)
            self.rows = []
            self.column = int(column)
            self.cryptofun = cryptofun
            self.bigrams = bigrams
            self.args = cryptfun_args
            self.retorno = []
            self.daemon = True
            self.lock = lock
            self.md5 = md5

    def set_rows(self, value):
        self.rows = value


    def get_column(self):
        return self.column


    def get_retorno(self):
        return self.retorno


    def run(self):
# TODO: impedir que a thread inicie sem setar as rows
        self.lock.acquire()
        for row in self.rows:
            original_data = row[self.column]
            if original_data == '':
                self.retorno.append(NullField("vazio"))
            else:
                if self.bigrams:
                    from lib.mybloom.bloomfilter import BloomFilter
                    bloomfilter1 = BloomFilter(self.args['size'],self.args['fp'])
                    index = ngram.NGram(N=2)
                    bigrams = list(index.ngrams(index.pad(str(original_data))))
                    
                    for bigram in bigrams:
                        bloomfilter1.add(str(bigram))
                    
                    self.retorno.append(bloomfilter1)
                elif self.md5:
                    self.retorno.append(hashlib.md5(original_data.encode()).hexdigest())
                else:
    #                 pk = self.args['pubkey'] 
    #                 m_data = self.cryptofun.encrypt(pk, int(float(original_data)))
    #                 m_data = self.cryptofun.encrypt(int(float(original_data)))
                    try:
#                         if original_data == '':
#                             d = float(-9999999999999999)
#                         else:
                        d = float(original_data)
                    except:
                        print(20*"=")
                        print(original_data)
                        import sys
                        sys.exit()
                    m_data = self.cryptofun.encrypt(d)
                    self.retorno.append(m_data)
        
        self.lock.release()
#         threadLock.release()


def encryptDataSet(data,cryptofuns,header=False):
    """
        Method to encrypt a dataset
        data : List of list (use csvutil.read)
        cryptofuns : List of EncryptRows
        
    """
    
    data_length = len(data)
    start = 0
    if not header:
        start = 1

    threads = []
    
    for fun in cryptofuns:
        fun.set_rows(data[1:])
        fun.start()
#         fun.join()
        threads.append(fun)

    for t in threads:
        t.join()
        
    for i in range(start,data_length):
        for fun in cryptofuns:
            data[i][fun.get_column()] = fun.get_retorno()[i-1]
            
    return data


def capply(data,colum,fun,bigram=False,header=False, **keys):
    data_length = len(data)
    start = 0
    if not header:
        start = 1
    
    for i in range(start,data_length):
        original_data = data[i][colum]
        # transform data in bigrams, bloomfilter
        if bigram:
            bloomfilter1 = fun(keys['size'],keys['fp'])
            index = ngram.NGram(N=2)
            bigrams = list(index.ngrams(index.pad(original_data)))
            
            for bigram in bigrams:
                bloomfilter1.add(str(bigram))
            
            data[i][colum] = bloomfilter1
        else:
#             pk = keys['pubkey'] 
#             m_data = fun.encrypt(pk, int(float(original_data)) )
            d = float(original_data)
            m_data = fun.encrypt(d)
            
            data[i][colum] = m_data

    return data


def convertDateStr2Date(ncol,data):
    
    import datetime
    #import time
    #a = time.mktime(datetime.datetime.strptime("01/01/1811", "%d/%m/%Y").timetuple())
    for i in range(1,len(data)):
        
        try:
            v = datetime.datetime.strptime(data[i][ncol], "%d/%m/%Y")
        except ValueError:
            v = datetime.datetime.strptime("01/01/0001", "%d/%m/%Y")
        
        data[i][ncol] = v
        
    return data

def convertDateStr2AgeInDays(ncol,data):
    
    import datetime
    #import time
    #a = time.mktime(datetime.datetime.strptime("01/01/1811", "%d/%m/%Y").timetuple())
    now = datetime.datetime.now()
    
    for i in range(1,len(data)):
        
        try:
            v = datetime.datetime.strptime(data[i][ncol], "%d/%m/%Y")
        except ValueError:
            try:
                v = datetime.datetime.strptime(data[i][ncol], "%d-%m-%Y")
            except ValueError:
                v = datetime.datetime.strptime("01/01/0001", "%d/%m/%Y")
        
        age = now - v
        data[i][ncol] = age.days
        
    return data
if __name__ == '__main__':
    import time
    from util.csvutil import *
    #file = "C:\Users\Thiago\Dropbox\Mestrado\workspace\python\dptee4RL\data\micro_data.csv"
    file = "/media/sf_Mestrado/workspace/python/dptee4RL/data/micro_data.csv"
    
    
    start_m1 = time.time()
    from util.csvutil import *
    
    od = read(file,headers=True)
    rows = convertDateStr2AgeInDays(7,od)
    rows = convertDateStr2AgeInDays(8,rows)
    for r in rows:
        print(r[7],r[8])
    import sys
    sys.exit()

    
#     b = paillier.e_mul_const(pub, pd[2][4], -1)
#     r = paillier.e_add(pub, pd[1][4], pd[2][4])
#     paillier.decrypt(priv, pub, r)
#     
#     r = paillier.e_add(pub, md[1][4], md[2][4])
#     print(paillier.decrypt(priv, pub, r))
#     
#     r = paillier.e_add(pub, pd[1][4], md[2][4])
#     print(paillier.decrypt(priv, pub, r))
# m1 exec time :  139.621999979
# m2 exec time :  97.5850000381