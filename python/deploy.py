import time
from web3 import Web3, HTTPProvider

web3 = Web3(HTTPProvider('http://localhost:8545'))

contract_name = 'CC1'

#read contract data
#bin_dir = 'D:\\Dados\\OneDrive\\Doutorado\\workspace\\bc-playground\\bin\\'
bin_dir = '/home/thiagonobrega/workspace/bc-playground/bin/'
c_abi = open(bin_dir + contract_name + '.abi', "r").read()
c_bin = open(bin_dir + contract_name + '.bin', "r").read()

#create contract
AIFMRMCoin = web3.eth.contract(abi=c_abi, bytecode=c_bin)

#unlock account
web3.personal.unlockAccount(web3.eth.accounts[1], '', 0)

#deploy contract
tx_hash = AIFMRMCoin.constructor().transact({'from': web3.eth.accounts[1], 'gas': 1000000})

# Get tx receipt to get contract address
tx_receipt = web3.eth.getTransactionReceipt(tx_hash)

contract_address = tx_receipt['contractAddress']
contract_address
'0x52DB67A188a2ddAd8433A80C494CbBb15002D125'

cc = web3.eth.contract(
    address=contract_address,
    abi=c_abi,
)


cc.functions.compareBloom(bytes([1]),bytes([3])).transact()
transaction={'from': web3.eth.accounts[0], 'gas': 1000000, 'to': contract_address}

cc.functions.compareBloom(bytes([1]),bytes([3])).call(transaction)



## Uso do contrato
# Mint 500 coins and check balance
#AIFMRMCoin.transact(transaction={'from': web3.eth.accounts[0], 'gas': 1000000, 'to': contract_address}).mint(web3.eth.accounts[0], 500)
#AIFMRMCoin.call(transaction={'from': web3.eth.accounts[0], 'gas': 1000000, 'to': contract_address}).balances(web3.eth.accounts[0])