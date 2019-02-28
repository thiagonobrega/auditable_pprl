pragma solidity >0.4.99 <0.6.0;

contract ComparasionClassification {

    
     /** @dev Jaccard distance over to BloomFiltes
      * @param bf1 Width of the rectangle.
      * @param bf2 Height of the rectangle.
      * @return distance * 10**precison
      */
    function jaccardBloom(uint256 bf1, uint256 bf2,uint precision) internal pure returns (uint256) {
        uint inter = countBits(uint256(bf1 & bf2));
        uint union = countBits(uint256(bf1 | bf2));
        
        if (inter == 0 && union == 0){
            return 1 * precision * 10;
        }

        // require(bytes(inter).length > 0);
        require(inter != 0,"Intercption <= 0");
        require(union != 0,"Union <= 0");
        
        
        
        uint q = inter*(10**precision);
        uint r = q/union;

        // inter*(10**precision)/union
        return r;
    }

    /** @dev Count the number of bits in a uint256 
      * @param callnum the uint256
      * @return the number of ones
      */
    function countBits(uint256 callnum) internal pure returns (uint) {
        uint count = 0;
        uint256 num = callnum; //fixe later

        while (num != 0){
            num = num & (num-1);
            count = count + 1;
        }

        return count;
    }

    function compareEntities(uint256 bf1, uint256 bf2) public pure returns (uint256) {
        uint precision = 2;
        return jaccardBloom(bf1,bf2,precision);
    }

}