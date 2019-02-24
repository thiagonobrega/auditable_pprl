pragma solidity >0.4.99 <0.6.0;

contract CC1 {

    function compareBloom(bytes1 bf1, bytes1 bf2) public payable returns (bytes1) {
        return bf1 & bf2;
    }

    function jaccardBloom(bytes1 bf1, bytes1 bf2,uint precision) public payable returns (uint256) {
        // uint a = ;
        // b = uint(bf1 | bf2);
        uint inter = countBits(uint8(bf1 & bf2));
        uint union = countBits(uint8(bf1 | bf2));
        return inter*(10**precision)/union;
    }

    function countBits(uint num) public payable returns (uint) {
        uint count = 0;

        while (num != 0){
            num = num & (num-1);
            count = count + 1;
        }

        return count;
    }
}