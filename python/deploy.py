import time
from web3 import Web3, HTTPProvider
from solcx import compile_files, link_code , compile_standard
from solcx import get_installed_solc_versions, set_solc_version

# def readCompiledContractData(contract_name,web3,file_path='/home/thiagonobrega/workspace/bc-playground/bin/'):
#     web3=Web3(HTTPProvider('http://localhost:8545'))
#     #read contract data
#     #bin_dir = 'D:\\Dados\\OneDrive\\Doutorado\\workspace\\bc-playground\\bin\\'
#     bin_dir = file_path
#     c_abi = open(bin_dir + contract_name + '.abi', "r").read()
#     c_bin = open(bin_dir + contract_name + '.bin', "r").read()

#     #create contract
#     return web3.eth.contract(abi=c_abi, bytecode=c_bin)

def get_input_json(file_path,file,lib,ldlib):
    # https://solidity.readthedocs.io/en/v0.5.3/using-the-compiler.html
    if lib == None:
        input_json =  {
                    "language": "Solidity",
                    "sources": {
                                    str(file): {
                                    "urls": [
                                        str(file_path+file)
                                    ]
                                    }
                                },
                    "settings": {
                                    "optimizer": {
                                        "enabled": True,
                                        "runs": 200
                                    },
                                    "outputSelection": {
                                                str(file) : {
                                                            "*": [
                                                            "metadata",
                                                            "abi",
                                                            "evm.bytecode",
                                                            "evm.sourceMap"]
                                                            }
                                    }
                                }
                    
        }
    else:
        input_json =  {
                    "language": "Solidity",
                    "sources": {
                                    str(file): {
                                    "urls": [
                                        str(file_path+file)
                                    ]
                                    },
                                    str(lib): {
                                    "urls": [
                                        str(file_path+lib)
                                    ]
                                    }
                                },
                    "settings": {
                                    "optimizer": {
                                        "enabled": True,
                                        "runs": 200
                                    },
                                    "metadata": {
                                        "useLiteralContent": True
                                    },
                                    "libraries": {
                                        str(file): {
                                        str(list(ldlib.keys())[0]):str(ldlib[list(ldlib.keys())[0]])
                                        }
                                    },
                                    "outputSelection": {
                                                str(file) : {
                                                            "*": [
                                                            "metadata",
                                                            "abi",
                                                            "evm.bytecode",
                                                            "evm.sourceMap"]
                                                            }
                                    }
                                }
                    
        }
    return input_json
                    

def compileContract(file,lib=None,ldlib=None,file_path="/home/thiagonobrega/workspace/bc-playground/Contracts/"):

    input_json = get_input_json(file_path,file,lib,ldlib)
    set_solc_version('v0.5.4')
    # return compile_files([file_path+file])
    return compile_standard(input_json, allow_paths=file_path)

def deployContract(account,file,contract,contract_name,web3,library_address=None):
    # unlocck account
    web3.personal.unlockAccount(account,'', 0)

    cbytecode = contract['contracts'][file][contract_name]['evm']['bytecode']['object']
    cabi = contract['contracts'][file][contract_name]['abi']

    # Link Contract with code
    if library_address != None:
        cbytecode = link_code(cbytecode, library_address)

    instanceContract = web3.eth.contract(abi=cabi,
                                        bytecode=cbytecode)
    
    # caso seja necessario utilizar o compilado
    # instanceContract = readCompiledContractData(contract_name,web3)

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
    sfile = 'pprl_lib.sol'
    compC = compileContract(sfile)
    
    libName = 'ComparisonTool'

    cc = deployContract(account,sfile,compC,libName,web3)
    print(cc)

    library_address = {
        str(libName): str(cc.address)
    }

    # library_address = {
    #     "stringUtils.sol:StringUtils": deploy_contract(library_link)
    # }

    sfile = 'cc3.sol'
    compC = compileContract(sfile,lib='pprl_lib.sol',ldlib=library_address)
    cName = 'ComparasionClassification'
    

    
    # cbytecode = compC['contracts'][sfile][cName]['evm']['bytecode']['object']
    # cbytecode = link_code(cbytecode, library_address)
    # cn = deployContract(account,'cc2.sol',compC,cName,web3,library_address=library_address)
    
    cn = deployContract(account,'cc3.sol',compC,cName,web3)

    #transaction
    web3.personal.unlockAccount(account,'', 0)
    transaction={'from': account, 'gas': 1000000, 'to': cn.address}
    cn.functions.compareEntities(bytes([1]),bytes([3])).call(transaction)
    cn.functions.compareEntities(bytes([1]),bytes([3])).transact(transaction)
    sys.exit(10)

    # first contract version
    print("Start deploying V1")
    sfile = 'cc.sol'
    compC = compileContract(sfile)

    cName = 'ComparasionClassification'

    cc = deployContract(account, sfile, compC, cName, web3)

    web3.personal.unlockAccount(account, '', 0)

    estimated_gas = cc.functions.compareEntities(bytes([1]), bytes([3])).estimateGas()


    transaction = {'from': account, 'gas': estimated_gas * 100, 'to': cc.address}


# //lista[1][1] [1][3]
    bf1 = int([1796462574145599402897438328675413491728501682111757066863021392605688588261313524111693578991849793617116431838277285716444340715873681589454313701831280645724974685926691938026572487671431429249457077933456669184535152673613475765362480743133340037512211648389783546318089385079287481706286468671440959540753939571674881414739356066682360929299140884390031053986436416])
    bf2 = int([1417339412015488780066050747435102632161478804817840413710209614188088266842068443593449269061577617159237509457013026941209910964486107061116509447317178157642633861512055661724526049722369461539986889302520475084166300371483023165748615314915778361267684316103501087541211295044117701622687720941172250905116051017901412598692788546243656116367388273818448008888232262])
    bf1 = 1796462574145599402897438328675413491728501682111757066863021392605688588261313524111693578991849793617116431838277285716444340715873681589454313701831280645724974685926691938026572487671431429249457077933456669184535152673613475765362480743133340037512211648389783546318089385079287481706286468671440959540753939571674881414739356066682360929299140884390031053986436416
    bf2 = 1417339412015488780066050747435102632161478804817840413710209614188088266842068443593449269061577617159237509457013026941209910964486107061116509447317178157642633861512055661724526049722369461539986889302520475084166300371483023165748615314915778361267684316103501087541211295044117701622687720941172250905116051017901412598692788546243656116367388273818448008888232262
    cc.functions.compareEntities(web3.toBytes(bf1), web3.toBytes(bf2)).call(transaction)



#     sds
    block = web3.eth.getBlock("latest");
    block.gasLimit




# 



#unlock account




# Get tx receipt to get contract address



# contract_address = tx_receipt['contractAddress']
# # contract_address
# # '0x52DB67A188a2ddAd8433A80C494CbBb15002D125'

# cc = web3.eth.contract(
#     address=contract_address,
#     abi=c_abi,
# )
# transaction={'from': web3.eth.accounts[0], 'gas': 1000000, 'to': contract_address}

# cc.functions.jaccardBloom(bytes([1]),bytes([3]),int(4)).call(transaction)

# print ('Creator',cc.call().creator)
# print ('Contracts',cc.call().newContracts)
# print ('OracleName',cc.call().oracleName)

# cc.functions.compareBloom(bytes([1]),bytes([3])).transact()
# transaction={'from': web3.eth.accounts[0], 'gas': 1000000, 'to': contract_address}
# cc.functions.compareBloom(bytes([1]),bytes([3])).call(transaction)
# cc.functions.compareBloom(bytes([1]),bytes([3])).transact(transaction)

# cc.functions.countBits(int(1)).call(transaction)



# ## Uso do contrato
# # Mint 500 coins and check balance
# #AIFMRMCoin.transact(transaction={'from': web3.eth.accounts[0], 'gas': 1000000, 'to': contract_address}).mint(web3.eth.accounts[0], 500)
# #AIFMRMCoin.call(transaction={'from': web3.eth.accounts[0], 'gas': 1000000, 'to': contract_address}).balances(web3.eth.accounts[0])