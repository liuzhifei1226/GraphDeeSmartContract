pragma solidity ^0.4.4;

contract owned {
    address public owner;

    function owned() public {
        owner = msg.sender;
    }

    modifier onlyOwner {
        require(msg.sender == owner);
        _;
    }

    function transferOwnership(address newOwner) onlyOwner public {
        owner = newOwner;
    }
}

contract Token {

    function totalSupply() public constant returns (uint256 supply) {}

    function balanceOf(address _owner) public constant returns (uint256 balance) {}

    function transfer(address _to, uint256 _value) public returns (bool success) {}

    function transferFrom(address _from, address _to, uint256 _value) public returns (bool success) {}

    function approve(address _spender, uint256 _value) public returns (bool success) {}

    function allowance(address _owner, address _spender) public constant returns (uint256 remaining) {}

    event Transfer(address indexed _from, address indexed _to, uint256 _value);
    event Approval(address indexed _owner, address indexed _spender, uint256 _value);

}

contract StandardToken is owned, Token {

    function transfer(address _to, uint256 _value) onlyOwner public returns (bool success) {
        //if (balances[msg.sender] >= _value && balances[_to] + _value > balances[_to]) {
        if (balances[msg.sender] >= _value && _value > 0) {
            balances[msg.sender] -= _value;
            balances[_to] += _value;
            Transfer(msg.sender, _to, _value);
            return true;
        } else { return false; }
    }

    function transferFrom(address _from, address _to, uint256 _value) onlyOwner public returns (bool success) {
        if (balances[_from] >= _value && allowed[_from][msg.sender] >= _value && _value > 0) {
            balances[_to] += _value;
            balances[_from] -= _value;
            allowed[_from][msg.sender] -= _value;
            Transfer(_from, _to, _value);
            return true;
        } else { return false; }
    }

    function balanceOf(address _owner) public constant returns (uint256 balance) {
        return balances[_owner];
    }

    function approve(address _spender, uint256 _value) public returns (bool success) {
        allowed[msg.sender][_spender] = _value;
        Approval(msg.sender, _spender, _value);
        return true;
    }

    function allowance(address _owner, address _spender) public constant returns (uint256 remaining) {
      return allowed[_owner][_spender];
    }

    mapping (address => uint256) balances;
    mapping (address => mapping (address => uint256)) allowed;
    uint256 public totalSupply;
}

contract ZGT is owned, StandardToken {

    /* Public variables of the token */

    string public name;
    uint8 public decimals;
    string public symbol;
    string public version = 'H1.0';
    uint256 public unitsOneEthCanBuy;
    uint256 public totalEthInWei;
    address public fundsWallet;
    uint256 public total_supply;

    // This is a constructor function
    function ZGT(uint256 total_tokens, string tokenName, string Symbol, uint256 oneTokenPrice) public {
        total_supply = total_tokens * 10 ** uint256(18);
        balances[msg.sender] = total_supply;
        totalSupply = total_supply;
        name = tokenName;
        decimals = 18;
        symbol = Symbol;
        unitsOneEthCanBuy = oneTokenPrice;
        fundsWallet = msg.sender;
    }


    function changeOwnerWithTokens(address newOwner) onlyOwner public {
        owner = newOwner;
        balances[owner] = balances[fundsWallet];
        balances[fundsWallet] = 0;
        fundsWallet = owner;
    }

    function changePrice(uint256 _newAmount) onlyOwner public {
      unitsOneEthCanBuy = _newAmount;
    }


    function() public payable {
        totalEthInWei = totalEthInWei + msg.value;
        uint256 amount = msg.value * unitsOneEthCanBuy;
        if (balances[fundsWallet] < amount) {
            return;
        }

        balances[fundsWallet] = balances[fundsWallet] - amount;
        balances[msg.sender] = balances[msg.sender] + amount;

        Transfer(fundsWallet, msg.sender, amount); // Broadcast a message to the blockchain

        //Transfer ether to fundsWallet
        fundsWallet.transfer(msg.value);
    }

    /* Approves and then calls the receiving contract */
    function approveAndCall(address _spender, uint256 _value, bytes _extraData) public returns (bool success) {
        allowed[msg.sender][_spender] = _value;
        Approval(msg.sender, _spender, _value);

        if(!_spender.call(bytes4(bytes32(sha3("receiveApproval(address,uint256,address,bytes)"))), msg.sender, _value, this, _extraData)) { revert(); }
        return true;
    }
}
