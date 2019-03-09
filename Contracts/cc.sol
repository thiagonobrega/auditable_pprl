pragma solidity >0.4.99 <0.6.0;

contract ComparasionClassification {

    function jaccardBloom(bytes32 bf1, bytes32 bf2,uint precision) internal pure returns (uint256) {
        uint inter = countBits(bf1 & bf2);
        uint union = countBits(bf1 | bf2);
        
        if (inter == 0 && union == 0){
            return 1 * precision * 10;
        }

        // require(bytes(inter).length > 0);
        require(inter != 0,"Intercption <= 0");
        require(union != 0,"Union <= 0");

        //bytes memory a = new bytes(10);
        //bytes memory b = new bytes(10);
        //a & b;        
        
        uint q = inter*(10**precision);
        uint r = q/union;

        // inter*(10**precision)/union
        return r;
    }

    
    function countBits(bytes32 callnum) internal pure returns (uint) {
        uint count = 0;
        uint256 num = uint256(callnum); //fixe later

        while (num != 0){
            num = num & (num-1);
            count = count + 1;
        }

        return count;
    }

    function compareEntities(bytes32 bf1, bytes32 bf2) public pure returns (uint256) {
        uint precision = 2;
        return jaccardBloom(bf1,bf2,precision);
    }

}