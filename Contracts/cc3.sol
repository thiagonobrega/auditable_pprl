pragma solidity >0.4.99 <0.6.0;

import "./pprl_lib.sol";

/** @title Shape calculator. */
contract ComparasionClassification {
    using ComparisonTool for *;

     /** @dev Calculates a rectangle's surface and perimeter.
      * @param bf1 Width of the rectangle.
      * @param bf2 Height of the rectangle.
      * @return s The calculated surface.
      */
    function compareEntities(bytes32 bf1, bytes32 bf2) public pure returns (uint256) {
        // require(otherContract.doSomething()); // where it returns a bool
        uint precision = 3;
        return ComparisonTool.jaccardBloom(bf1,bf2,precision);
    }

}