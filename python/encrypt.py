'''
Created on 9 de abr de 2017

@author: Thiago
'''


import csv
import os
import time
import datetime

import pandas as pd
import bitarray
import ngram
from web3 import Web3, HTTPProvider

from lib.util.bc import *
from lib.mybloom.bloomfilter import BloomFilter
from lib.mybloom.bloomutil import *
from deploy import compileContract
from lib.util.env import getbase_dir

def encryptData(data,size,fp=0.01,n=2):
    """
        n : 2 = Bigrams
        size : Size of BF
        fp : False positive rate
    """
    bloomfilter = BloomFilter(size,fp)
    index = ngram.NGram(N=n)
    bigrams = list(index.ngrams(index.pad(str(data))))

    for bigram in bigrams:
        bloomfilter.add(str(bigram))

    return bloomfilter

# '0xC1844bbe0537cE51F95F9EC08c55D697fCcf3f17'
def bitarray2int(ba):
    import struct
    return int(ba.to01(), 2)

def int2bitarray(num):
    return bitarray.bitarray(num)

def convertBF2BC_slow(bf):
    one = bytes(True)
    zero = bytes(False)
    bffilter = []

    for i in bf.filter:
        if( bytes(i) == one):
            bffilter.append(1)
        else:
            bffilter.append(0)

    # bytes(bffilter)
    return bytes(bffilter)
def convertBF2BC(bf):
    return bytes(bf.filter)
# end encrypt

def encrypt_data(datadir,basename,bflen,ngrams=2):

    base_dir = getbase_dir('Datasets') + datadir + os.sep

    rows = []
    # print(base_dir+basename)
    with open(base_dir + basename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                # print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                dbf1 = row[3] + row[4] + row[7]
                dbf2 = row[8] + row[9] + row[12]
                erow = [row[1], encryptData(dbf1, bflen, n=ngrams), row[2], encryptData(dbf2, bflen, n=ngrams)]
                rows.append(erow)
                line_count += 1
        # print(f'Processed {line_count} lines.')
    return rows

def exec_comparasion_regular_vs_bcjaccard(rows,web3,dc,sleep_time=0.5):
    results = [['id_l','id_r','regular_jaccard','exec_time_regular','bc_jaccard','calc_bcjaccard','exec_time_bc','estimated_gas']]

    row_count = 0
    for row in rows:
        # print(row_count)
        row_count += 1
        print(row_count,len(rows))
        bcbf1= convertBF2BC(row[1])
        bcbf2= convertBF2BC(row[3])

        try:
            estimated_gas = dc.functions.compareEntities(bcbf1, bcbf2).estimateGas()
            # print("\t Well estimated")
        except ValueError:
            estimated_gas = 100000
            # print("\t Not Well estimated")

        transaction = {'from': account, 'gas': estimated_gas +1, 'to': dc.address}

        start_bc = time.time()
        try:
            i,u , j = dc.functions.compareEntities(bcbf1, bcbf2).call()
            bcjac = j
            calc_bcjac = i/u
        except Exception as e:
            print(e)
            bcjac = -1
            calc_bcjac = -1
        end_bc = time.time()

        start_rj = time.time()
        regularjac = jaccard_coefficient(row[1],row[3])
        end_rj = time.time()

        id_l = row[0]
        id_r = row[2]
        r = [id_l,id_r,regularjac,end_rj-start_rj,bcjac,calc_bcjac,end_bc-start_bc,estimated_gas]
        results.append(r)
        time.sleep(sleep_time)
    return results

# import pandas as pd
# r = exec_comparasion_regular_vs_bcjaccard(encrypted_entities[0:5],web3,deployed_contract,bitlen=128,sleep_time=0.7)
# df = pd.DataFrame.from_records(r[1:], columns=r[0])
# print(df[['regular_jaccard' , 'bc_jaccard']])
# del r , df



# reogranizar em outro local
def get_DeployedContract(web3, contract_name, contract_address, contract_srcfile):
    # cdir = getbase_dir('Contracts')
    cc = compileContract(contract_srcfile)
    abi = cc['contracts'][contract_srcfile][contract_name]['abi']

    dc = web3.eth.contract(
        address=contract_address,
        abi=abi,
    )
    return dc

def save2csv(data,epath,file_name,bpath='..'+os.sep+'results'+os.sep):
    os.makedirs(bpath+epath, exist_ok=True)
    headers = data[0]
    df = pd.DataFrame.from_records(data[1:], columns=headers)
    df.to_csv(bpath+epath+os.sep+file_name, sep=';', encoding='utf-8')


if __name__ == 'main':
    print("Anonnymizing entities..")
    encrypted_entities = encrypt_data('bikes', 'candset.csv', 96)
    print("Setup BC-Connection")
    # web3 = Web3(HTTPProvider('http://localhost:8545'))

    #### compiling contract
    # regular
    web3 = Web3(HTTPProvider('http://localhost:9545'))
    # PoA
    # web3 = configure_poa_env(Web3(HTTPProvider('http://localhost:9545')))


    cFile = 'ccb.sol'
    cName = 'ComparasionClassification'
    account = web3.eth.accounts[1]

    print("Compiling SmartContracts")
    cc = compileContract(cFile)
    print("Deploying SmartContracts")
    from deploy import deployContract
    deployed_contract = deployContract(account, cFile, cc, cName, web3)

    #### alredy dployed contract
    # contract_address = '0xA76068c461716d34499cA221A037Cedb39067e26'
    # deployed_contract = get_DeployedContract(web3, 'ComparasionClassification', contract_address, 'ccb.sol')



    comparison_results = exec_comparasion_regular_vs_bcjaccard(encrypted_entities[0:5],web3,deployed_contract,sleep_time=0.01)
    # import pandas as pd
    # r = exec_comparasion_regular_vs_bcjaccard(encrypted_entities[0:5],web3,deployed_contract,bitlen=128,sleep_time=0.7)
    df = pd.DataFrame.from_records(comparison_results[1:], columns=comparison_results[0])
    print(df[['regular_jaccard' , 'bc_jaccard']])
    # del r , df


    # debug
    bf1 = encrypted_entities[0][1]
    bf2 = encrypted_entities[0][3]

    bcbf1 = convertBF2BC(bf1)
    bcbf2 = convertBF2BC(bf2)

    dc = deployed_contract
    estimated_gas = dc.functions.compareEntities(bcbf1, bcbf2).estimateGas()
    print(estimated_gas)
    estimated_gas = dc.functions.interUnion2(bcbf1, bcbf2).estimateGas()
    print(estimated_gas)
    account = web3.eth.accounts[1]
    transaction = {'from': account, 'gas': estimated_gas + 2000, 'to': dc.address}

    dc.functions.interUnion2(bcbf1, bcbf1).call(transaction)
    dc.functions.compareEntities(bcbf1, bcbf1).call(transaction)
    dc.functions.jaccardBloom(bcbf1, bcbf1, int(2)).call(transaction)



    dc.functions.getBFlen(bcbf1).call(transaction)
    i = dc.functions.interBf(bcbf1,bcbf2).call(transaction)
    u = dc.functions.unionBf(bcbf1, bcbf2).call(transaction)
    dc.functions.countBits(bytes(i)).call(transaction)/dc.functions.countBits(bytes(u)).call(transaction)


    ###
    #infura
    from web3 import Web3, HTTPProvider
    from deploy import createInstaceContract
    import time

    w3 = Web3(HTTPProvider("https://ropsten.infura.io/v3/ae955aa167d1467b94a59729f7b5770d"))
    cFile = 'ccb.sol'
    cName = 'ComparasionClassification'
    account = '0xfCBb70E6983e2a24c7Ee392D598A3060bAE95096'
    print("Compiling SmartContracts")
    cc = createInstaceContract(w3,cFile,cName)

    print("Deploying SmartContracts")
    from deploy import deployContractInfura
    smc = deployContractInfura(w3,cc,cFile,cName)

    print("Anonnymizing entities..")
    encrypted_entities = encrypt_data('bikes', 'candset.csv', 96)

    bf1 = encrypted_entities[0][1]
    bf2 = encrypted_entities[0][3]

    bcbf1 = convertBF2BC(bf1)
    bcbf2 = convertBF2BC(bf2)

    privateKey = 'A0AC3B5859D530FF6D43E06F36A24F9F2C3F12F47AC8EB87EEE9F058139753D3'
    acct = w3.eth.account.privateKeyToAccount(privateKey)

    # estimated_gas = dc.functions.compareEntities(bcbf1, bcbf2).estimateGas()
    # print(estimated_gas)
    estimated_gas = smc.functions.interUnion2(bcbf1, bcbf2).estimateGas()

    #### Transacao
    ##mostrar isso
    txn = smc.functions.interUnion2(bcbf1, bcbf2).buildTransaction()
    txn['nonce'] = w3.eth.getTransactionCount(acct.address)

    # mesmo resultado
    # signed_tx = w3.eth.account.signTransaction(txn, privateKey)
    signed_tx = acct.signTransaction(txn)

    result = w3.eth.sendRawTransaction(signed_tx.rawTransaction)

    tx_receipt = w3.eth.getTransactionReceipt(result)

    count = 0
    while tx_receipt is None and (count < 30):
        time.sleep(5)

        tx_receipt = w3.eth.getTransactionReceipt(result)

        print(tx_receipt)

    if tx_receipt is None:
        print("Falhei")
        # return {'status': 'failed', 'error': 'timeout'}


    ### call

    a,b = smc.functions.interUnion2(bcbf1, bcbf2).call()
    a,b = smc.functions.compareEntities(bcbf1, bcbf1).call()

    start = datetime.datetime.today().strftime('%d-%m-%Y_%H-%M-%S')
    r = exec_comparasion_regular_vs_bcjaccard(encrypted_entities,w3,smc,sleep_time=0.001)
    end = datetime.datetime.today().strftime('%d-%H-%M-%S')

    # r = exec_comparasion_regular_vs_bcjaccard(encrypted_entities[0:10], w3, smc, sleep_time=0.01)
    # df = pd.DataFrame.from_records(r[1:], columns=r[0])
    # print(df[['regular_jaccard', 'bc_jaccard']])
    file_name = 'bike-'+start+'-to-'+end+'.csv'
    save2csv(r,'e1',file_name)











