pragma solidity >0.4.99 <0.6.0;

contract CC1 {

//     function someOldFunction(uint8 a) external;
//     function anotherOldFunction() external returns (bool);
    
//     function doSomething(OldContract a) public returns (bool) {
//       a.someOldFunction(0x42);
//       return a.anotherOldFunction();
//    }

    function compareBloom(bytes1 bf1, bytes1 bf2) public pure returns (bytes1) {
        return bf1 & bf2;
    }


}