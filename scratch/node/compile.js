const path = require('path');
const fs = require('fs');
const solc = require('solc');

const contractPath = path.resolve(__dirname, '../Contracts', 'cc.sol');

let contractSource = fs.readFileSync(contractPath, 'utf-8');
//console.log("contract content: \n" + contractSource);

let jsonContractSource = JSON.stringify({
    language: 'Solidity',
    sources: {
      'Task': {
          content: contractSource,
       },
    },
    settings: { 
        outputSelection: {
            '*': {
                '*': ['abi',"evm.bytecode"],   
             // here point out the output of the compiled result
            },
        },
    },
});

let solcResult = solc.compile(jsonContractSource);
var output = JSON.parse(solcResult)
//console.log(output)
var contractName = 'CC1'
var deployable_contract = output.contracts.Task[contractName]

var dc_abi = deployable_contract.abi
var dc_bytecode = deployable_contract.evm

console.log(dc_abi);
console.log("estou aqui")
console.log(dc_bytecode);

//solcResult
//console.log(solcResult);
// console.log(solc.compile(source,1));


//deploy 
const Web3 = require("web3");
let ethnet_addr = 'http://localhost:8545';

web3 = new Web3(new Web3.providers.HttpProvider(ethnet_addr));

web3.utils.asciiToHex('Top 10 Students')
const newContract = web3.eth.contract(JSON.parse(interface))
  const options = { from: web3.eth.accounts[0], data: bytecode, gas: 1000000 }

newContract.new(options, newContractCallback(name))

function deployContract(contract){
    let abi = contract.abi
    let bytecode = contract.evm.bytecode.object

    let tokenContract = web3.eth.contract(abi);
    let contractData = null;

    // Prepare the smart contract deployment payload
    // If the smart contract constructor has mandatory parameters, you supply the input parameters like below 
    //
    // contractData = tokenContract.new.getData( param1, param2, ..., {
    //    data: '0x' + bytecode
    // });
    contractData = tokenContract.new.getData({
        data: '0x' + bytecode
    });   

}

function deployContractOriginal(contract) {

    // It will read the ABI & byte code contents from the JSON file in ./build/contracts/ folder
    let jsonOutputName = path.parse(contract).name + '.json';
    let jsonFile = './build/contracts/' + jsonOutputName;

    // After the smart deployment, it will generate another simple json file for web frontend.
    let webJsonFile = './www/assets/contracts/' + jsonOutputName;
    let result = false;

    try {
        result = fs.statSync(jsonFile);
    } catch (error) {
        console.log(error.message);
        return false;
    }

    // Read the JSON file contents
    let contractJsonContent = fs.readFileSync(jsonFile, 'utf8');    
    let jsonOutput = JSON.parse(contractJsonContent);

    // Retrieve the ABI 
    let abi = jsonOutput['contracts'][contract][path.parse(contract).name]['abi'];

    // Retrieve the byte code
    let bytecode = jsonOutput['contracts'][contract][path.parse(contract).name]['evm']['bytecode']['object'];
    
    let tokenContract = web3.eth.contract(abi);
    let contractData = null;

    // Prepare the smart contract deployment payload
    // If the smart contract constructor has mandatory parameters, you supply the input parameters like below 
    //
    // contractData = tokenContract.new.getData( param1, param2, ..., {
    //    data: '0x' + bytecode
    // });    

    contractData = tokenContract.new.getData({
        data: '0x' + bytecode
    });   

    // Prepare the raw transaction information
    let rawTx = {
        nonce: nonceHex,
        gasPrice: gasPriceHex,
        gasLimit: gasLimitHex,
        data: contractData,
        from: accounts[selectedAccountIndex].address
    };

    // Get the account private key, need to use it to sign the transaction later.
    let privateKey = new Buffer(accounts[selectedAccountIndex].key, 'hex')

    let tx = new Tx(rawTx);

    // Sign the transaction 
    tx.sign(privateKey);
    let serializedTx = tx.serialize();

    let receipt = null;

    // Submit the smart contract deployment transaction
    web3.eth.sendRawTransaction('0x' + serializedTx.toString('hex'), (err, hash) => {
        if (err) { 
            console.log(err); return; 
        }
    
        // Log the tx, you can explore status manually with eth.getTransaction()
        console.log('Contract creation tx: ' + hash);
    
        // Wait for the transaction to be mined
        while (receipt == null) {

            receipt = web3.eth.getTransactionReceipt(hash);

            // Simulate the sleep function
            Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 1000);
        }

        console.log('Contract address: ' + receipt.contractAddress);
        console.log('Contract File: ' + contract);

        // Update JSON
        jsonOutput['contracts'][contract]['contractAddress'] = receipt.contractAddress;

        // Web frontend just need to have abi & contract address information
        let webJsonOutput = {
            'abi': abi,
            'contractAddress': receipt.contractAddress
        };

        let formattedJson = JSON.stringify(jsonOutput, null, 4);
        let formattedWebJson = JSON.stringify(webJsonOutput);

        //console.log(formattedJson);
        fs.writeFileSync(jsonFile, formattedJson);
        fs.writeFileSync(webJsonFile, formattedWebJson);

        console.log('==============================');
    
    });
    
    return true;
}

// // env nao  funfa
// let gasPrice = web3.eth.gasPrice;
// let gasPriceHex = web3.toHex(gasPrice);
// let gasLimitHex = web3.toHex(6000000);
// let block = web3.eth.getBlock("latest");
// let nonce =  web3.eth.getTransactionCount(accounts[selectedAccountIndex].address, "pending");
// let nonceHex = web3.toHex(nonce);

//lixo

//
//var simpleStorage = StoreContract.new(your_parameters ,{from:web3.eth.accounts[0], data: compiled_code , gas: gas_amount}, call_back_function)



// var _name = /* var of type bytes32 here */ ;

// //abi
// var ownedtokenContract = web3.eth.contract([{"constant":false,
//                                             "inputs":[{"name":"newOwner","type":"address"}],
//                                             "name":"transfer","outputs":[],"payable":false,
//                                             "stateMutability":"nonpayable","type":"function"}
//                                             ,{"constant":false,"inputs":[{"name":"newName","type":"bytes32"}],"name":"changeName","outputs":[],"payable":false,"stateMutability":"nonpayable","type":"function"},
//                                             {"inputs":[{"name":"_name","type":"bytes32"}],"payable":false,"stateMutability":"nonpayable","type":"constructor"}]);

// var ownedtoken = ownedtokenContract.new(
//    _name,
//    {
//      from: web3.eth.accounts[0], 
//     //bytecode.object
//      data: '0x608060405234801561001057600080fd5b5060405160208061042e8339810180604052602081101561003057600080fd5b810190808051906020019092919050505033600160006101000a81548173ffffffffffffffffffffffffffffffffffffffff021916908373ffffffffffffffffffffffffffffffffffffffff160217905550336000806101000a81548173ffffffffffffffffffffffffffffffffffffffff021916908373ffffffffffffffffffffffffffffffffffffffff1602179055508060028190555050610355806100d96000396000f3fe608060405260043610610046576000357c0100000000000000000000000000000000000000000000000000000000900480631a6952301461004b578063898855ed1461009c575b600080fd5b34801561005757600080fd5b5061009a6004803603602081101561006e57600080fd5b81019080803573ffffffffffffffffffffffffffffffffffffffff1690602001909291905050506100d7565b005b3480156100a857600080fd5b506100d5600480360360208110156100bf57600080fd5b81019080803590602001909291905050506102c9565b005b600160009054906101000a900473ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff163373ffffffffffffffffffffffffffffffffffffffff16141515610133576102c6565b6000809054906101000a900473ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff1663c3cee9c1600160009054906101000a900473ffffffffffffffffffffffffffffffffffffffff16836040518363ffffffff167c0100000000000000000000000000000000000000000000000000000000028152600401808373ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff1681526020018273ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff1681526020019250505060206040518083038186803b15801561024357600080fd5b505afa158015610257573d6000803e3d6000fd5b505050506040513d602081101561026d57600080fd5b8101908080519060200190929190505050156102c55780600160006101000a81548173ffffffffffffffffffffffffffffffffffffffff021916908373ffffffffffffffffffffffffffffffffffffffff1602179055505b5b50565b6000809054906101000a900473ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff163373ffffffffffffffffffffffffffffffffffffffff16141561032657806002819055505b5056fea165627a7a723058204b9a6faaea26c5b34ae066765afa7e8e7dfb39401bc15351679116e6d268f5b20029', 
//      gas: '4700000'
//    }, function (e, contract){
//     console.log(e, contract);
//     if (typeof contract.address !== 'undefined') {
//          console.log('Contract mined! address: ' + contract.address + ' transactionHash: ' + contract.transactionHash);
//     }
//  })


// // for (var contractName in output.contracts['test.sol']) {
// //     console.log(contractName + ': ' + output.contracts['test.sol'][contractName].evm.bytecode.object)
// // }

