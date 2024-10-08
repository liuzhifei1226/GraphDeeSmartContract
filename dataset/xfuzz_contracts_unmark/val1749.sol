pragma solidity ^0.4.4;

contract SafeMath {
  //internals

  function safeMul(uint a, uint b) internal returns (uint) {
    uint c = a * b;
    assert(a == 0 || c / a == b);
    return c;
  }

  function safeSub(uint a, uint b) internal returns (uint) {
    assert(b <= a);
    return a - b;
  }

  function safeAdd(uint a, uint b) internal returns (uint) {
    uint c = a + b;
    assert(c>=a && c>=b);
    return c;
  }

  function assert(bool assertion) internal {
    if (!assertion) throw;
  }
}

contract Token is SafeMath {

    /// @return total amount of tokens
    function totalSupply() constant returns (uint256 supply) {}

    /// @param _owner The address from which the balance will be retrieved
    /// @return The balance
    function balanceOf(address _owner) constant returns (uint256 balance) {}

    /// @notice send `_value` token to `_to` from `msg.sender`
    /// @param _to The address of the recipient
    /// @param _value The amount of token to be transferred
    /// @return Whether the transfer was successful or not
    function transfer(address _to, uint256 _value) returns (bool success) {}

    /// @notice send `_value` token to `_to` from `_from` on the condition it is approved by `_from`
    /// @param _from The address of the sender
    /// @param _to The address of the recipient
    /// @param _value The amount of token to be transferred
    /// @return Whether the transfer was successful or not
    function transferFrom(address _from, address _to, uint256 _value) returns (bool success) {}

    /// @notice `msg.sender` approves `_addr` to spend `_value` tokens
    /// @param _spender The address of the account able to transfer the tokens
    /// @param _value The amount of wei to be approved for transfer
    /// @return Whether the approval was successful or not
    function approve(address _spender, uint256 _value) returns (bool success) {}

    /// @param _owner The address of the account owning tokens
    /// @param _spender The address of the account able to transfer the tokens
    /// @return Amount of remaining tokens allowed to spent
    function allowance(address _owner, address _spender) constant returns (uint256 remaining) {}

    event Transfer(address indexed _from, address indexed _to, uint256 _value);
    event Approval(address indexed _owner, address indexed _spender, uint256 _value);
    
    event Burned(uint amount);
}

contract StandardToken is Token {

    function transfer(address _to, uint256 _value) returns (bool success) {
        //Default assumes totalSupply can't be over max (2^256 - 1).
        //If your token leaves out totalSupply and can issue more tokens as time goes on, you need to check if it doesn't wrap.
        //Replace the if with this one instead.
        //if (balances[msg.sender] >= _value && balances[_to] + _value > balances[_to]) {
        if (now < icoEnd + lockedPeriod && msg.sender != fundsWallet) throw;
        if (msg.sender == fundsWallet && now < icoEnd + blockPeriod && ownerNegTokens < _value) throw; //prevent the owner of spending his share of tokens within the first year
        if (balances[msg.sender] >= _value && _value > 0) {
            balances[msg.sender] -= _value;
            balances[_to] += _value;
            Transfer(msg.sender, _to, _value);
            if (msg.sender == fundsWallet && now < icoEnd + blockPeriod) {
                ownerNegTokens = safeSub(ownerNegTokens, _value);
            }
            return true;
        } else { return false; }
    }

    function transferFrom(address _from, address _to, uint256 _value) returns (bool success) {
        //same as above. Replace this line with the following if you want to protect against wrapping uints.
        //if (balances[_from] >= _value && allowed[_from][msg.sender] >= _value && balances[_to] + _value > balances[_to]) {
        if (now < icoEnd + lockedPeriod && msg.sender != fundsWallet) throw;
        if (msg.sender == fundsWallet && now < icoEnd + blockPeriod && ownerNegTokens < _value) throw;
        if (balances[_from] >= _value && allowed[_from][msg.sender] >= _value && _value > 0) {
            balances[_to] += _value;
            balances[_from] -= _value;
            allowed[_from][msg.sender] -= _value;
            Transfer(_from, _to, _value);
            if (msg.sender == fundsWallet && now < icoEnd + blockPeriod) {
                ownerNegTokens = safeSub(ownerNegTokens, _value);
            }
            return true;
        } else { return false; }
    }

    function balanceOf(address _owner) constant returns (uint256 balance) {
        return balances[_owner];
    }

    function approve(address _spender, uint256 _value) returns (bool success) {
        allowed[msg.sender][_spender] = _value;
        Approval(msg.sender, _spender, _value);
        return true;
    }

    function allowance(address _owner, address _spender) constant returns (uint256 remaining) {
      return allowed[_owner][_spender];
    }
    
    function burn(){
    	//if tokens have not been burned already and the ICO ended
    	if(!burned && now> icoEnd){
    		uint256 difference = tokensToSell;//checked for overflow above
    		balances[fundsWallet] = balances[fundsWallet] - difference;
    		totalSupply = totalSupply - difference;
    		burned = true;
    		Burned(difference);
    	}
    }

    mapping (address => uint256) balances;
    mapping (address => mapping (address => uint256)) allowed;
    uint256 public totalSupply;
    
    uint256 public icoStart = 1520244000;
    
    uint256 public icoEnd = 1520244000 + 45 days;
    
    //ownerFreezeTokens tokens will be freezed during this period after ICO
    uint256 public blockPeriod = 1 years;
    
    //after this period after ICO end token holders can operate with them
    uint256 public lockedPeriod = 15 days;
    
    //owners negotiable token that he can spend in any time
    uint256 public ownerNegTokens = 13500000000000000000000000;
    
    //owner tokens to be feezed on year
    uint256 public ownerFreezeTokens = 13500000000000000000000000;
    
    //max number of tokens that can be sold
    uint256 public tokensToSell = 63000000000000000000000000; 
    
    bool burned = false;
    
    string public name;                   
    uint8 public decimals = 18;                
    string public symbol;                 
    string public version = 'H1.0'; 
    uint256 public unitsOneEthCanBuy;     
    uint256 public totalEthInWei = 0;          
    address public fundsWallet;
}

contract EpsToken is StandardToken {

    // This is a constructor function 
    // which means the following function name has to match the contract name declared above
    function EpsToken() {
        balances[msg.sender] = 90000000000000000000000000;              
        totalSupply = 90000000000000000000000000;                     
        name = "Epsilon";                                            
        symbol = "EPS";                                             
        unitsOneEthCanBuy = 28570;                                      
        fundsWallet = msg.sender;                         
    }

    function() payable{
        
        if (now < icoStart || now > icoEnd || tokensToSell <= 0) {
            return;
        }
        
        totalEthInWei = totalEthInWei + msg.value;
        uint256 amount = msg.value * unitsOneEthCanBuy;
        uint256 valueInWei = msg.value;
        
        if (tokensToSell < amount) {
            amount = tokensToSell;
            valueInWei = amount / unitsOneEthCanBuy;
            msg.sender.transfer(msg.value - valueInWei);
        }
        
        tokensToSell -= amount;

        balances[fundsWallet] = balances[fundsWallet] - amount;
        balances[msg.sender] = balances[msg.sender] + amount;
        
        
        Transfer(fundsWallet, msg.sender, amount); // Broadcast a message to the blockchain

        //Transfer ether to fundsWallet
        fundsWallet.transfer(valueInWei);                               
    }

    /* Approves and then calls the receiving contract */
    function approveAndCall(address _spender, uint256 _value, bytes _extraData) returns (bool success) {
        allowed[msg.sender][_spender] = _value;
        Approval(msg.sender, _spender, _value);

        //call the receiveApproval function on the contract you want to be notified. This crafts the function signature manually so one doesn't have to include a contract in here just for this.
        //receiveApproval(address _from, uint256 _value, address _tokenContract, bytes _extraData)
        //it is assumed that when does this that the call *should* succeed, otherwise one would use vanilla approve instead.
        if(!_spender.call(bytes4(bytes32(sha3("receiveApproval(address,uint256,address,bytes)"))), msg.sender, _value, this, _extraData)) { throw; }
        return true;
    }
}
