pragma solidity >0.4.99 <0.6.0;

contract TestLib {

    
    uint internal constant ZERO = uint(0);
    uint internal constant ONE = uint(1);
    uint internal constant ONES = uint(~0);
    bytes32 internal constant B32_ZERO = bytes32(0);
    bytes1 internal constant btsNull = bytes1(0);
    
    /**
     * Length
     * 
     * Returns the length of the specified string
     * 
     * @param _base When being used for a data type this is the extended object
     *              otherwise this is the string to be measured
     * @return uint The length of the passed string
     */
    function length(string memory _base)  internal  pure  returns (uint) {
        bytes memory _baseBytes = bytes(_base);
        return _baseBytes.length;
    }
    
    /**
     * Index Of
     *
     * Locates and returns the position of a character within a string starting
     * from a defined offset
     * 
     * @param _base When being used for a data type this is the extended object
     *              otherwise this is the string acting as the haystack to be
     *              searched
     * @param _value The needle to search for, at present this is currently
     *               limited to one character
     * @param _offset The starting point to start searching from which can start
     *                from 0, but must not exceed the length of the string
     * @return int The position of the needle starting from 0 and returning -1
     *             in the case of no matches found
     */
    function _indexOf(string memory _base, string memory _value, uint _offset) public pure returns (int) {
        bytes memory _baseBytes = bytes(_base);
        bytes memory _valueBytes = bytes(_value);

        assert(_valueBytes.length == 1);

        for(uint i = _offset; i < _baseBytes.length; i++) {
            if (_baseBytes[i] == _valueBytes[0]) {
                return int(i);
            }
        }

        return -1;
    }

    /**
     * Sub String
     * 
     * Extracts the part of a string based on the desired length and offset. The
     * offset and length must not exceed the lenth of the base string.
     * 
     * @param _base When being used for a data type this is the extended object
     *              otherwise this is the string that will be used for 
     *              extracting the sub string from
     * @param _length The length of the sub string to be extracted from the base
     * @param _offset The starting point to extract the sub string from
     * @return string The extracted sub string
     */
    function _substring(string memory _base, int _length, int _offset) public pure returns (string memory) {
        bytes memory _baseBytes = bytes(_base);

        assert(uint(_offset+_length) <= _baseBytes.length);

        string memory _tmp = new string(uint(_length));
        bytes memory _tmpBytes = bytes(_tmp);

        uint j = 0;
        for(uint i = uint(_offset); i < uint(_offset+_length); i++) {
          _tmpBytes[j++] = _baseBytes[i];
        }

        return string(_tmpBytes);
    }


    /**
     * Parse Int
     * 
     * Converts an ASCII string value into an uint as long as the string 
     * its self is a valid unsigned integer
     * 
     * @param _value The ASCII string to be converted to an unsigned integer
     * @return uint The unsigned value of the ASCII string
     */
    function parseInt(string memory _value) 
        public pure
        returns (uint _ret) {
        bytes memory _bytesValue = bytes(_value);
        uint j = 1;
        for(uint i = _bytesValue.length-1; i >= 0 && i < _bytesValue.length; i--) {
            bytes1 z = _bytesValue[i];
            uint8 a;
            assembly { mstore(add(a, 8), z) }
            
            assert( a >= 48 && a <= 57);
            _ret += (uint(a) - 48)*j;
            j*=10;
        }
    }
    
    function stringToUint(string memory s) public pure returns (uint) {
        bool hasError = false;
        bytes memory b = bytes(s);
        uint result = 0;
        uint oldResult = 0;
        for (uint i = 0; i < b.length; i++) { // c = b[i] was not needed
            bytes1 z = b[i];
            if ( uint8(z) >= 48 && uint8(z) <= 57) {
                // store old value so we can check for overflows
                oldResult = result;
                result = result * 10 + ( uint8(z) - 48); // bytes and int are not compatible with the operator -.
                // prevent overflows
                if(oldResult > result ) {
                    // we can only get here if the result overflowed and is smaller than last stored value
                    hasError = true;
                }
            } else {
                hasError = true;
            }
        }
        return result; 
    }
    
    function t1(string memory base, string memory delimiter,uint8 num_of_parts) public pure returns (uint[] memory){
        uint pos = 0;
        int dp = 0;
        uint[] memory out = new uint[](num_of_parts);
        int[] memory sout = new int[](num_of_parts);

        uint8 cpart = 0;

        dp = _indexOf(base, delimiter , pos);
        sout[cpart] = dp;
        
        while (dp != -1){
            int llength = dp - int(pos);
            string memory temps =_substring(base, llength, int(pos));
            out[cpart] = stringToUint(temps);
            sout[cpart] = dp;
            
            pos = uint(dp+1);
            cpart++;
            dp = _indexOf(base, delimiter , pos);
        }
        
        int llength = int(length(base)) - int(pos);
        string memory temps =_substring(base, llength, int(pos));
        out[cpart] = stringToUint(temps);
        
        return out;
    }
    
    
    // converted
    /**
     * Int conversions
     */
    function uint2562Bytes(uint256 x) public pure returns (bytes memory b) {
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
    
    
    /**
     * string2bytes32
     *
     * Locates and returns the position of a character within a string starting
     * from a defined offset
     * 
     * @param base When being used for a data type this is the extended object
     *              otherwise this is the string acting as the haystack to be
     *              searched
     * @param delimiter The needle to search for, at present this is currently
     *               limited to one character
     * @param num_of_parts The starting point to start searching from which can start
     *                from 0, but must not exceed the length of the string
     * @param num_of_records The starting point to start searching from which can start
     *                from 0, but must not exceed the length of the string
     * @return bytes32[] The position of the needle starting from 0 and returning -1
     *             in the case of no matches found
     */
    function string2bytes32(string memory base, string memory delimiter,string memory string_delimiter, uint8 num_of_parts , uint8 num_of_records) public pure returns (bytes32[] memory){
        
        uint8 xxx = num_of_parts * num_of_records;
        
        // uint8 l = num_of_records * num_of_parts;
        bytes32[] memory out = new bytes32[](xxx);
        uint pos = 0;
        int dp = 0;
        
        //bytes[] memory sout = new bytes[](num_of_records*num_of_parts);
        

        // get the first record separator position
        for (uint j = 0; j < num_of_records; j++){
            
            dp = _indexOf(base, string_delimiter , pos);
            
            int llength;
            if (dp != -1){
                llength = dp - int(pos);
                // return (int(pos),llength);
            } else {
                llength = int(length(base)) - int(pos);
            }
            
            string memory temps =_substring(base, llength, int(pos)); // extract the first record
            uint[] memory bf_parts = t1(temps, delimiter, num_of_parts);
            // adicona os bytes ao array de bytes
            
            for (uint i = 0; i < num_of_parts ; i++){
                
                uint p = (j*num_of_parts) + i;
                
                uint256 x = bf_parts[i];
                bytes32 b = bytes32(x);
        
                require(out.length >= p, "1 - ERROR: tamanho do >=");
                // require(out.length > record, "1 - ERROR: tamanho do >");
                out[p] = b; 
                
            }
            
            // atualiza a posicao
            if (dp != -1){
                pos = uint(dp)+1;
            }
            
        }
        
        return out;
    }
    
    
    /* bytes32 (fixed-size array) to bytes (dynamically-sized array) */
    function bytes32ToBytes(bytes32 _bytes32) public pure returns (bytes memory){
        // string memory str = string(_bytes32);
        // TypeError: Explicit type conversion not allowed from "bytes32" to "string storage pointer"
        bytes memory bytesArray = new bytes(32);
        for (uint256 i; i < 32; i++) {
            bytesArray[i] = _bytes32[i];
        }
        return bytesArray;
    }//
    
    function bfAssembler(bytes32[] memory rdata,uint index, uint8 num_of_parts) public pure returns (bytes memory){
        
        // bytes memory bf = new bytes(num_of_parts*32);
        // uint index_bf = 0;
        bytes memory bf;
        
        for (uint i = 0; i < num_of_parts; i++) {
            
            uint pos = (index * num_of_parts) + i;
            if (i == 0){
                bf = bytes32ToBytes(rdata[pos]);
                // return bf;
            } else {
                bytes memory part = bytes32ToBytes(rdata[pos]);
                bf = concatBytes(bf,part);
            }
            
            if (i == num_of_parts -1){
                return bf;
            }
            
        }
        
        return bf;
        // bytes memory bf = uint2562Bytes(bf_parts[0]);
    }
    
    function tbf(uint index,string memory result) public pure returns (bytes memory){
        uint8 num_of_parts = 4;
        uint8 num_of_records = 2;
        bytes32[] memory rdata = string2bytes32(result,"#","!",num_of_parts,num_of_records);
        return bfAssembler(rdata, index, num_of_parts);
    }
    
    
    // function fixme(string memory base, string memory delimiter,uint8 num_of_parts) public pure returns (bytes memory){

    //     uint[] memory bf_parts = t1(base, delimiter, num_of_parts);
        
    //     bytes memory bf = uint2562Bytes(bf_parts[0]);
        
    //     for (uint i = 1; i < num_of_parts; i++) {
    //         bytes memory part = uint2562Bytes(bf_parts[i]);
    //         bf = concatBytes(bf,part);
    //     }
        
    //     return bf;
    // }

    

}