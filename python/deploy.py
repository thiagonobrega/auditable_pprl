import time
from web3 import Web3, HTTPProvider

web3 = Web3(HTTPProvider('http://localhost:8545'))

contract_name = 'MyToken'

#read contract data
bin_dir = 'D:\\Dados\\OneDrive\\Doutorado\\workspace\\bc-playground\\bin\\'
c_abi = open(bin_dir + contract_name + '.abi', "r").read()
c_bin = open(bin_dir + contract_name + '.bin', "r").read()

#create contract
AIFMRMCoin = web3.eth.contract(abi=c_abi, bytecode=c_bin)

#unlock account
web3.personal.unlockAccount(web3.eth.accounts[13], 'thiago', 0)

#deploy contract
tx_hash = AIFMRMCoin.constructor().transact({'from': web3.eth.accounts[13], 'gas': 1000000})

# Get tx receipt to get contract address
tx_receipt = web3.eth.getTransactionReceipt(tx_hash)

contract_address = tx_receipt['contractAddress']
contract_address

## Uso do contrato
# Mint 500 coins and check balance
#AIFMRMCoin.transact(transaction={'from': web3.eth.accounts[0], 'gas': 1000000, 'to': contract_address}).mint(web3.eth.accounts[0], 500)
#AIFMRMCoin.call(transaction={'from': web3.eth.accounts[0], 'gas': 1000000, 'to': contract_address}).balances(web3.eth.accounts[0])