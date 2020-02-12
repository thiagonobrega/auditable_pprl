# Instruction 

In order to, to execute the Blockchain experiments, we need to ;
 - Configure a Blockchain
 - Deploy a web service to simulate the PPRL participants
 - Deploy the Semi-Trusted Third Party Smart Contract 



## Blockchain 

The experiments can be executed in two scenarios; i) in a private Ethereum network or ii) in the public Ethereum network. 

To execute the private Ethereum network, we recommend the use of a docker environment to build the test netwok. A detailed tutorial is provided in this [repository](https://github.com/Capgemini-AIE/ethereum-docker). 

To use the public Ethereum network, follow the instruction provided in this [tutorial](https://medium.com/swlh/deploy-smart-contracts-on-ropsten-testnet-through-ethereum-remix-233cd1494b4b).


### Web Service

We provide an example of web service (implemented in node.js) to simulate the PPRL participants.  This web service must be deployed in a server with a public IP; that is, the smart contract must access the web service. The instructions to deploy the web service is provided in the node directory of this project. 

### Semi-Trusted Third Party Smart Contract





https://steemit.com/json/@chrisdotn/a-json-parser-for-solidity


https://ethereum.stackexchange.com/questions/6121/parse-json-in-solidity
