import "./oraclizeAPI.sol";
pragma solidity >= 0.5.0 < 0.6.0; 

contract ExampleContract is usingOraclize {

   string public ETHUSD;
   event LogConstructorInitiated(string nextStep);
   event LogPriceUpdated(string price);
   event LogNewOraclizeQuery(string description);

   constructor () public {
       emit LogConstructorInitiated("Constructor was initiated. Call 'updatePrice()' to send the Oraclize Query.");
   }

   function __callback(bytes32 myid, string memory result) public {
       if (msg.sender != oraclize_cbAddress()) revert();
       ETHUSD = result;
       emit LogPriceUpdated(result);
   }

   function updatePrice() public payable {
       if (oraclize_getPrice("URL") > address(this).balance) {
           emit LogNewOraclizeQuery("Oraclize query was NOT sent, please add some ETH to cover for the query fee");
       } else {
           emit LogNewOraclizeQuery("Oraclize query was sent, standing by for the answer..");
           oraclize_query("URL", "json(https://api.pro.coinbase.com/products/ETH-USD/ticker).price");
       }
   }
} 