pragma solidity >0.4.99 <0.6.0;

library ComparisonTool {
    /**
     * Count the number of bits in one uint256 bits
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

    /**
    * Jaccard distance over to BloomFilter
     */
    function jaccardBloom(bytes32 bf1, bytes32 bf2,uint precision) public pure returns (uint256) {
        uint inter = countBits(uint256(bf1 & bf2));
        uint union = countBits(uint256(bf1 | bf2));
        return inter*(10**precision)/union;
    }

}

contract ComparasionClassification {
    using ComparisonTool for *;

    function compareEntities(bytes32 bf1, bytes32 bf2) public pure returns (uint256) {
        uint precision = 3;
        return ComparisonTool.jaccardBloom(bf1,bf2,precision);
    }

}