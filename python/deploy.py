import time
from web3 import Web3, HTTPProvider



def readCompiledContractData(contract_name,web3,file_path='/home/thiagonobrega/workspace/bc-playground/bin/'):
    web3=Web3(HTTPProvider('http://localhost:8545'))
    #read contract data
    #bin_dir = 'D:\\Dados\\OneDrive\\Doutorado\\workspace\\bc-playground\\bin\\'
    bin_dir = file_path
    c_abi = open(bin_dir + contract_name + '.abi', "r").read()
    c_bin = open(bin_dir + contract_name + '.bin', "r").read()

    #create contract
    return web3.eth.contract(abi=c_abi, bytecode=c_bin)

def deployContract(account,contract_name,web3):
    # unlocck account
    web3.personal.unlockAccount(account,'', 0)

    instanceContract = readCompiledContractData(contract_name,web3)
    #deploy contract
    tx_hash = instanceContract.constructor().transact({'from': account, 'gas': 1000000})

    tx_receipt = web3.eth.getTransactionReceipt(tx_hash)
    iter_count = 1

    while tx_receipt == None or iter_count < 30:
        print("Trying.. " + str(iter_count) + "/30 ..." )
        time.sleep(2)
        tx_receipt = web3.eth.getTransactionReceipt(tx_hash)
        iter_count+=1

    if tx_receipt == None:
        import sys
        print("Aborting!!!")
        sys.exit()
    
    contract_address = tx_receipt['contractAddress']
    # contract_address
    # '0x52DB67A188a2ddAd8433A80C494CbBb15002D125'

    cc = web3.eth.contract(
        address=contract_address,
        abi=instanceContract.abi,
    )

    return cc



if __name__ == "__main__":
    import sys
    web3=Web3(HTTPProvider('http://localhost:8545'))
    account = web3.eth.accounts[1]
    print("Start deploying lib")
    
    cname = 'ComparisonTool'
    cc = deployContract(account,cname,web3)
    print(cc)
    '0x90Dd5486A271235Be9508f62AcEBfd275D6e6404'


    sys.exit(10)

transaction={'from': account 'gas': 1000000, 'to': contract_address}



#unlock account




# Get tx receipt to get contract address



contract_address = tx_receipt['contractAddress']
# contract_address
# '0x52DB67A188a2ddAd8433A80C494CbBb15002D125'

cc = web3.eth.contract(
    address=contract_address,
    abi=c_abi,
)
transaction={'from': web3.eth.accounts[0], 'gas': 1000000, 'to': contract_address}

cc.functions.jaccardBloom(bytes([1]),bytes([3]),int(4)).call(transaction)

print ('Creator',cc.call().creator)
print ('Contracts',cc.call().newContracts)
print ('OracleName',cc.call().oracleName)

cc.functions.compareBloom(bytes([1]),bytes([3])).transact()
transaction={'from': web3.eth.accounts[0], 'gas': 1000000, 'to': contract_address}

cc.functions.compareBloom(bytes([1]),bytes([3])).call(transaction)
cc.functions.compareBloom(bytes([1]),bytes([3])).transact(transaction)

cc.functions.countBits(int(1)).call(transaction)



## Uso do contrato
# Mint 500 coins and check balance
#AIFMRMCoin.transact(transaction={'from': web3.eth.accounts[0], 'gas': 1000000, 'to': contract_address}).mint(web3.eth.accounts[0], 500)
#AIFMRMCoin.call(transaction={'from': web3.eth.accounts[0], 'gas': 1000000, 'to': contract_address}).balances(web3.eth.accounts[0])