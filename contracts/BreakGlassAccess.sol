// Local-testnet reference contract for the research prototype.
// It is NOT an audited medical access contract.
pragma solidity ^0.8.20;
contract BreakGlassAccess {
    address public owner;
    mapping(address => bool) public licensed;
    event EmergencyAccessRequested(address indexed doctor, bytes32 proofHash);
    event EmergencyAccessGranted(address indexed doctor, uint256 timestamp);
    constructor(){ owner=msg.sender; }
    modifier onlyOwner(){ require(msg.sender==owner,"owner only"); _; }
    function setLicensed(address doctor,bool ok) external onlyOwner { licensed[doctor]=ok; }
    function requestEmergencyAccess(bytes32 proofHash) external {
        emit EmergencyAccessRequested(msg.sender,proofHash);
        require(licensed[msg.sender],"not licensed");
        emit EmergencyAccessGranted(msg.sender,block.timestamp);
    }
}
