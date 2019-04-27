pragma solidity >0.4.99 <0.6.0;

contract ComparasionClassification {

    
    uint internal constant ZERO = uint(0);
    uint internal constant ONE = uint(1);
    uint internal constant ONES = uint(~0);
    bytes32 internal constant B32_ZERO = bytes32(0);
    bytes1 internal constant btsNull = bytes1(0);

    function interBf(uint bflen, bytes memory bf1,bytes memory bf2) public pure returns (bytes memory){
        
        bytes memory inter = new bytes(bflen);

        for (uint i = 0; i < bflen; i++){
            if (bf1[i] == bf2[i]){
                inter[i] = bf1[i];
            }
        }

        return inter;
    }

    function unionBf(uint bflen, bytes memory bf1,bytes memory bf2) public pure returns (bytes memory){
        
        bytes memory union = new bytes(bflen);

        for (uint i = 0; i < bflen; i++){
            if (bf1[i] != btsNull) {
                union[i] = bf1[i];
            }
            if (bf2[i] != btsNull) {
                union[i] = bf2[i];
            }
        }

        return union;
    }


    /** @dev Jaccard distance over to BloomFiltes
      * @param bflen length of bloomfilter
      * @param bf1 Width of the rectangle.
      * @param bf2 Height of the rectangle.
      * @return distance * 10**precison
      */
    function jaccardBloom(uint bflen,bytes memory bf1, bytes memory bf2,uint precision) internal pure returns (uint256) {
        // uint inter = countBits(bf1);
        uint inter = countBits(bflen, interBf(bflen, bf1, bf2));
        uint union = countBits(bflen, unionBf(bflen, bf1, bf2));
        
        if (inter == 0 && union == 0){
            return 1 * precision * 10;
        }

        // require(bytes(inter).length > 0);
        require(inter != 0,"Intercption <= 0");
        require(union != 0,"Union <= 0");
        
        // uint q = inter*(10**precision);
        // uint r = q/union;
        // inter*(10**precision)/union
        return inter*(10**precision)/union;
    }

    /** @dev Count the number of bits in a uint256 
      * @param bflen the uint256
      * @param bf the uint256
      * @return the number of ones
      */
    function countBits(uint bflen, bytes memory bf) public pure returns (uint) {
        uint count = 0;

        for (uint i = 0; i < bflen; i++){
            if (bf[i] != btsNull){
                count++;
            }
        }
        

        return count;
    }

    function compareEntities(uint bflen, bytes memory bf1, bytes memory bf2) public pure returns (uint256) {
        uint precision = 2;
        return jaccardBloom(bflen,bf1,bf2,precision);
    }

}