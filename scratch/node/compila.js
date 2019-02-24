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

//console.log(dc_abi);
//console.log("estou aqui")
//console.log(dc_bytecode);


//deploy 
console.log("Deploy")
const Web3 = require("web3");
let ethnet_addr = 'http://localhost:8545';

//
const HDWalletProvider = require("truffle-hdwallet-provider");

// Create a wallet provider to connect outside rinkeby network
const provider = new HDWalletProvider(mnemonic, accessToken, 1);

web3 = new Web3(new Web3.providers.HttpProvider(ethnet_addr));

0x007ccffb7916f37f7aeef05e8096ecfbe55afc2f

let abi = deployable_contract.abi
let bytecode = deployable_contract.evm.bytecode.object

let myContract = web3.eth.Contract(abi);
const options = { from: web3.eth.accounts[0], data: '0x' + bytecode , gas: 1000000 }

console.log(web3.eth.accounts[0])
console.log(options)

//myContract.deploy(options)

// var name = "CC1"
// myContract.new(options, newContractCallback(name))
// we3.eth.accounts[0]