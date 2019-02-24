pragma solidity >0.4.99 <0.6.0;

contract ComparasionClassification {

    /**
    * Jaccard distance over to BloomFiltes
    * @param bf1 
    * @param bf2 
    * 
    * @return distance * 10**precison
     */
    function jaccardBloom(bytes32 bf1, bytes32 bf2,uint precision) public payable returns (uint256) {
        uint inter = countBits(uint256(bf1 & bf2));
        uint union = countBits(uint256(bf1 | bf2));
        return inter*(10**precision)/union;
    }

    /**
     * Count the number of bits in one uint256 bits
     */
    function countBits(uint256 callnum) public payable returns (uint) {
        uint count = 0;
        uint256 num = callnum; //fixe later

        while (num != 0){
            num = num & (num-1);
            count = count + 1;
        }

        return count;
    }

}