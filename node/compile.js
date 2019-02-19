const path = require('path');
const fs = require('fs');
const solc = require('solc');

const contractPath = path.resolve(__dirname, '../Contracts', 'cc.sol');

console.log("estou aqui")

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

console.log(solcResult);
// console.log(solc.compile(source,1));
