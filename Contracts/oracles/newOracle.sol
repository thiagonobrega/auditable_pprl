import "./oraclizeAPI.sol";
import "./mbf.sol";
import "./ccc.sol";

pragma solidity >= 0.5.0 < 0.6.0; 

contract AaContract is usingOraclize {

    string public ETHUSD;
    uint[] public similarity_;
    uint[] public inter_;
    uint[] public union_;

   address constant public libAddr = 0x5b80D287f64446068ae190490e2019946F92fb9c;
   address constant public cccAddr = 0x0ED0663F65Ff2fa4728219A507E36E34b9E39005;
   
//   0x6d58b3c21d0b61fe738f0a6a29e300b62d9ed48f;

   event LogConstructorInitiated(string nextStep);
   event LogPriceUpdated(string price);
   event LogNewOraclizeQuery(string description);
   
   
   event LogComparisonResult(string comp);
   
   
   event LogResult(int[]);

   constructor () public {
       emit LogConstructorInitiated("Constructor was initiated. Call 'updatePrice()' to send the Oraclize Query.");
   }
    
    function getContractBalance() public view returns (uint256) { //view amount of ETH the contract contains
        return address(this).balance;
    }

    function withdraw() public { //withdraw all ETH previously sent to this contract
        msg.sender.transfer(address(this).balance);
    }

   function __callback(bytes32 myid, string memory result) public {
        if (msg.sender != oraclize_cbAddress()) revert();

        TestLib w = TestLib(libAddr);
        ComparasionClassification ccc = ComparasionClassification(cccAddr);
        
        uint8 nparts = 2;
        uint8 records = 2;
        
        
        bytes32[] memory w32 = w.string2bytes32(result,"#","!",nparts,records);

        for (uint8 i = 0; i < records ; i++ ){
            for (uint8 j = 0; j < records ; j++ ){
                if (i != j && j < i){
                    bytes memory bf1 = w.bfAssembler(w32,i,nparts);
                    bytes memory bf2 = w.bfAssembler(w32,j,nparts);

                    (uint a,uint b, uint c) = ccc.jaccardBloom(bf1,bf2,2);

                    // if ( c >= threshold){
                    inter_.push(a);
                    union_.push(b);
                    similarity_.push(c);
                    // }
                    
                    // emit LogComparisonResult(appendUintToString("RC : ",c));
                }
                
            }
        }
        // emit LogPriceUpdated("passei do for");

    //   ETHUSD = result;
    //   emit LogPriceUpdated(result);
   }

   function getBlk(string memory blk_num) public payable returns( string memory) {
       if (oraclize_getPrice("URL") > address(this).balance) {
           emit LogNewOraclizeQuery("Oraclize query was NOT sent, please add some ETH to cover for the query fee");
       } else {
           emit LogNewOraclizeQuery("Oraclize query was sent, standing by for the answer..");
            string memory _prefix = "json(http://catfish.westus2.cloudapp.azure.com/api/getBlk/";
            string memory _sufix = ").filter";
            bytes memory _bprefix = bytes(_prefix);
            bytes memory _bbnum = bytes(blk_num);
            bytes memory _bsufix = bytes(_sufix);

            string memory url = new string(_bprefix.length + _bbnum.length +_bsufix.length);
            bytes memory burl = bytes(url);

            uint k = 0;
            for (uint i = 0; i < _bprefix.length; i++) burl[k++] = _bprefix[i];
            for (uint i = 0; i < _bbnum.length; i++) burl[k++] = _bbnum[i];
            for (uint i = 0; i < _bsufix.length; i++) burl[k++] = _bsufix[i];

            url = string(burl);


        oraclize_query("URL", url,7999999);
        // oraclize_query("URL", "json(http://catfish.westus2.cloudapp.azure.com/blk/"+"ss"+").price");
        //    catfish.westus2.cloudapp.azure.com

       }
   }
   
//   function appendUintToString(string memory inStr, uint v) public pure returns (string memory str) {
//         uint maxlength = 100;
//         bytes memory reversed = new bytes(maxlength);
//         uint i = 0;
//         while (v != 0) {
//             uint remainder = v % 10;
//             v = v / 10;
//             reversed[i++] = new bytes(48 + remainder);
//         }
//         bytes memory inStrb = bytes(inStr);
//         bytes memory s = new bytes(inStrb.length + i);
//         uint j;
//         for (j = 0; j < inStrb.length; j++) {
//             s[j] = inStrb[j];
//         }
//         for (j = 0; j < i; j++) {
//             s[j + inStrb.length] = reversed[i - 1 - j];
//         }
//         str = string(s);
//     }
    
    function getResult(uint256 index) public view returns (uint,uint,uint){
        return (inter_[index],union_[index],similarity_[index]);
    }

}