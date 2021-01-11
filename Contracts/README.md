# Requirements

| Requirement | Version |
| --- | --- |
| Solidity | 0.5 > |
| Oracalize | 5 > |


## Instruction 

In order to, to execute the Blockchain experiments, we need to ;
 - Configure a Blockchain
 - Deploy a web service to simulate the PPRL participants
 - Deploy the Smart Contract Semi-Trusted Third Party (SC-STTP)


### Blockchain 

The experiments can be executed in two scenarios; i) in a private Ethereum network or ii) in the public Ethereum network. 

To execute the private Ethereum network, we recommend the use of a docker environment to build the test netwok. A detailed tutorial is provided in this [repository](https://github.com/Capgemini-AIE/ethereum-docker). 

To use the public Ethereum network, follow the instruction provided in this [tutorial](https://medium.com/swlh/deploy-smart-contracts-on-ropsten-testnet-through-ethereum-remix-233cd1494b4b).


### Web Service

We provide an example of web service (implemented in node.js) to simulate the PPRL participants.  This web service must be deployed in a server with a public IP; that is, the smart contract must access the web service. The instructions to deploy the web service is provided in the node directory of this project. 

### Semi-Trusted Third Party Smart Contract

The SC-STTP is available at [scsttp.sol](./oracles/newOracle.sol). To deploy SC-STTP in the Ethereum network, we recommend the usage of the [Remix IDE](https://remix.ethereum.org/). Configure the remix according to the following [tutorial](https://medium.com/swlh/deploy-smart-contracts-on-ropsten-testnet-through-ethereum-remix-233cd1494b4b).

Before the deployment of the SC-STTP, it is needed to deploy the comparison lib [ccc.sol](./v2/ccc.sol) and the [libaddr.sol] (./v2/lib.sol). Save the address of theses libs and change the address of the libs in the SC-STTP;

```javascript
address constant public libAddr = 0x5b80D287f64446068ae190490e2019946F92fb9c;	 
address constant public cccAddr = 0x0ED0663F65Ff2fa4728219A507E36E34b9E39005;
```

It is important to change the URL of the web service in SC-STTP:

```javascript
string memory _prefix = "json(http://catfish.westus2.cloudapp.azure.com/api/getBlk/";
```

Deploy SC-STTP and invoke the getBlk().
