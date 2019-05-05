pragma solidity >0.4.99 <0.6.0;

contract ComparasionClassification {

    
    uint internal constant ZERO = uint(0);
    uint internal constant ONE = uint(1);
    uint internal constant ONES = uint(~0);
    bytes32 internal constant B32_ZERO = bytes32(0);
    bytes1 internal constant btsNull = bytes1(0);

    function getBFlen(bytes memory bf1) public pure returns (uint){
        return bf1.length;
    }

    function interBf(bytes memory bf1,bytes memory bf2) public pure returns (bytes memory){
        
        require(bf1.length == bf2.length,"Diferent length filters");
        uint bflen = bf1.length;
        bytes memory inter = new bytes(bflen);

        for (uint i = 0; i < bflen; i++){
            if (bf1[i] == bf2[i]){
                inter[i] = bf1[i];
            }
        }

        return inter;
    }

    function unionBf(bytes memory bf1,bytes memory bf2) public pure returns (bytes memory){
        
        require(bf1.length == bf2.length,"Diferent length filters");
        uint bflen = bf1.length;
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
      * @param bf1 Width of the rectangle.
      * @param bf2 Height of the rectangle.
      * @return distance * 10**precison
      */
    function jaccardBloom(bytes memory bf1, bytes memory bf2,uint precision) public pure returns (uint256, uint256 , uint256) {
        // uint inter = countBits(bf1);
        uint inter = countBits(interBf(bf1, bf2));
        uint union = countBits(unionBf(bf1, bf2));
        
        if (inter == 0 && union == 0){
            uint r = 1 * precision * 10;
            return (inter, union , r);
        }

        if (union == 0){
            //uint r = 1 * precision * 10;
            return (inter, union , 0);
        }

    
        uint q = inter*(10**precision);
        uint r = q/union;

        return (inter, union , r) ;
    }

    function interUnion1(bytes memory bf1, bytes memory bf2) public pure returns (uint256, uint256) {
        uint inter = countBits(interBf(bf1, bf2));
        uint union = countBits(unionBf(bf1, bf2));
        return (inter, union);
    }

    function interUnion2(bytes memory bf1, bytes memory bf2) public pure returns (uint256, uint256) {        
        require(bf1.length == bf2.length,"Diferent length filters");
        uint bflen = bf1.length;
        
        bytes memory union = new bytes(bflen);
        bytes memory inter = new bytes(bflen);

        for (uint i = 0; i < bflen; i++){
            if (bf1[i] != btsNull) {
                union[i] = bf1[i];
            }
            if (bf2[i] != btsNull) {
                union[i] = bf2[i];
            }
            if (bf1[i] == bf2[i]){
                inter[i] = bf1[i];
            }
        }

        return (countBits(inter), countBits(union));
    }

    /** @dev Count the number of bits in a uint256 
      * @param bf the uint256
      * @return the number of ones
      */
    function countBits(bytes memory bf) public pure returns (uint) {
        uint bflen = bf.length;
        uint count = 0;

        for (uint i = 0; i < bflen; i++){
            if (bf[i] != btsNull){
                count++;
            }
        }
        

        return count;
    }

    function compareEntities(bytes memory bf1, bytes memory bf2) public pure returns (uint256,uint256,uint256) {
        uint precision = 2;
        return jaccardBloom(bf1,bf2,precision);
    }


}