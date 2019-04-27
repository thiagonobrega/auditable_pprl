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


    // new methods
    function assembleBF(uint256 bflen) public pure returns (bytes memory) {
        bytes memory bf = new bytes(bflen);
        return bf;
    }

    function toBytes(uint256 x) public pure returns (bytes memory b) {
        b = new bytes(32);
        assembly { 
                    mstore(add(b, 32), x) 
        }
    }

    // https://github.com/GNSPS/solidity-bytes-utils/blob/master/contracts/BytesLib.sol
    function concatBytes( bytes memory preBytes, bytes memory postBytes) public pure returns (bytes memory) {
        bytes memory tempBytes;

        assembly {
            // Get a location of some free memory and store it in tempBytes as Solidity does for memory variables.
            tempBytes := mload(0x40)

            // Store the length of the first bytes array at the beginning of the memory for tempBytes.
            let length := mload(preBytes)
            mstore(tempBytes, length)

            // Maintain a memory counter for the current write location in the temp bytes array by adding the 32 bytes for the array length to
            // the starting location.
            let mc := add(tempBytes, 0x20)
            // Stop copying when the memory counter reaches the length of the first bytes array.
            let end := add(mc, length)

            for {
                // Initialize a copy counter to the start of the _preBytes data, 32 bytes into its memory.
                let cc := add(preBytes, 0x20)
            } lt(mc, end) {
                // Increase both counters by 32 bytes each iteration.
                mc := add(mc, 0x20)
                cc := add(cc, 0x20)
            } {
                // Write the _preBytes data into the tempBytes memory 32 bytes at a time.
                mstore(mc, mload(cc))
            }

            // Add the length of _postBytes to the current length of tempBytes and store it as the new length in the first 32 bytes of the
            // tempBytes memory.
            length := mload(postBytes)
            mstore(tempBytes, add(length, mload(tempBytes)))

            // Move the memory counter back from a multiple of 0x20 to the
            // actual end of the _preBytes data.
            mc := end
            // Stop copying when the memory counter reaches the new combined
            // length of the arrays.
            end := add(mc, length)

            for {
                let cc := add(postBytes, 0x20)
            } lt(mc, end) {
                mc := add(mc, 0x20)
                cc := add(cc, 0x20)
            } {
                mstore(mc, mload(cc))
            }

            // Update the free-memory pointer by padding our last write location
            // to 32 bytes: add 31 bytes to the end of tempBytes to move to the
            // next 32 byte block, then round down to the nearest multiple of
            // 32. If the sum of the length of the two arrays is zero then add 
            // one before rounding down to leave a blank 32 bytes (the length block with 0).
            mstore(0x40, and(
              add(add(end, iszero(add(length, mload(preBytes)))), 31),
              not(31) // Round down to the nearest 32 bytes.
            ))
        } // assembly

        return tempBytes;
    }

}