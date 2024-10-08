pragma solidity ^0.4.11;

/**
 * @title ETHCON Early Bird Donation Contract
 * @author majoolr.io
 *
 * Accepts donations and issues ETHCON token if at or above 3.9604 ETH.
 * See ETHCON.org for further information.
 * ETHCONEarlyBirdToken contract at 0x2d9498d0fd6f40760d53a847eb64eaf51c9b8e74
 */

contract ETHCONEarlyBirdDonation {
  address majoolr;
  ETHCONEarlyBirdToken token;

  uint256 public donations;
  mapping (address => uint256) public donationMap;
  mapping (address => uint256) public failedDonations;
  uint256 public minimum = 3960400000000000000;

  event ErrMsg(address indexed _from, string _msg);
  event ThxMsg(address indexed _from, string _msg);

  modifier andIsMajoolr {
    require(msg.sender == majoolr);
    _;
  }

  function(){ ErrMsg(msg.sender, 'No function called'); }

  function ETHCONEarlyBirdDonation(address _token){
    token = ETHCONEarlyBirdToken(_token);
    majoolr = msg.sender;
  }

  function donate() payable returns (bool){
    uint256 totalDonation = donationMap[msg.sender] + msg.value;
    if(totalDonation < minimum){
      failedDonations[msg.sender] += msg.value;
      ErrMsg(msg.sender, "Donation too low, call withdrawDonation()");
      return false;
    }

    bool success = token.transferFrom(majoolr,msg.sender,1);
    if(!success){
      failedDonations[msg.sender] += msg.value;
      ErrMsg(msg.sender, "Transer failed, call withdrawDonation()");
      return false;
    }

    donationMap[msg.sender] += msg.value;
    donations += msg.value;
    ThxMsg(msg.sender, "Thank you for your donation!");
    return true;
  }

  function generousDonation() payable returns (bool){
    uint256 tokensLeft = token.allowance(majoolr, this);
    if(tokensLeft == 0){
      failedDonations[msg.sender] += msg.value;
      ErrMsg(msg.sender, "No more donations here check Majoolr.io, call withdrawDonation()");
      return false;
    }

    donationMap[msg.sender] += msg.value;
    donations += msg.value;
    ThxMsg(msg.sender, "Thank you for your donation!");
    return true;
  }

  function withdraw() andIsMajoolr {
    uint256 amount = donations;
    donations = 0;
    msg.sender.transfer(amount);
  }

  function withdrawDonation(){
    uint256 amount = failedDonations[msg.sender];
    failedDonations[msg.sender] = 0;
    msg.sender.transfer(amount);
  }
}

contract ETHCONEarlyBirdToken {
   using ERC20Lib for ERC20Lib.TokenStorage;

   ERC20Lib.TokenStorage token;

   string public name = "ETHCON-Early-Bird";
   string public symbol = "THX";
   uint public decimals = 0;
   uint public INITIAL_SUPPLY = 600;

   event ErrorMsg(string msg);

   function ETHCONEarlyBirdToken() {
     token.init(INITIAL_SUPPLY);
   }

   function totalSupply() constant returns (uint) {
     return token.totalSupply;
   }

   function balanceOf(address who) constant returns (uint) {
     return token.balanceOf(who);
   }

   function allowance(address owner, address spender) constant returns (uint) {
     return token.allowance(owner, spender);
   }

   function transfer(address to, uint value) returns (bool ok) {
     if(token.balanceOf(to) == 0){
       return token.transfer(to, value);
     } else {
       ErrorMsg("Recipient already has token");
       return false;
     }

   }

   function transferFrom(address from, address to, uint value) returns (bool ok) {
     if(token.balanceOf(to) == 0){
       return token.transferFrom(from, to, value);
     } else {
       ErrorMsg("Recipient already has token");
       return false;
     }
   }

   function approve(address spender, uint value) returns (bool ok) {
     return token.approve(spender, value);
   }

   event Transfer(address indexed from, address indexed to, uint value);
   event Approval(address indexed owner, address indexed spender, uint value);
}

library ERC20Lib {
  using BasicMathLib for uint256;

  struct TokenStorage {
    mapping (address => uint256) balances;
    mapping (address => mapping (address => uint256)) allowed;
    uint totalSupply;
  }

  event Transfer(address indexed from, address indexed to, uint256 value);
  event Approval(address indexed owner, address indexed spender, uint256 value);
  event ErrorMsg(string msg);

  /// @dev Called by the Standard Token upon creation.
  /// @param self Stored token from token contract
  /// @param _initial_supply The initial token supply
  function init(TokenStorage storage self, uint256 _initial_supply) {
    self.totalSupply = _initial_supply;
    self.balances[msg.sender] = _initial_supply;
  }

  /// @dev Transfer tokens from caller's account to another account.
  /// @param self Stored token from token contract
  /// @param _to Address to send tokens
  /// @param _value Number of tokens to send
  /// @return success True if completed, false otherwise
  function transfer(TokenStorage storage self, address _to, uint256 _value) returns (bool success) {
    bool err;
    uint256 balance;

    (err,balance) = self.balances[msg.sender].minus(_value);
    if(err) {
      ErrorMsg("Balance too low for transfer");
      return false;
    }
    self.balances[msg.sender] = balance;
    //It's not possible to overflow token supply
    self.balances[_to] = self.balances[_to] + _value;
    Transfer(msg.sender, _to, _value);
    return true;
  }

  /// @dev Authorized caller transfers tokens from one account to another
  /// @param self Stored token from token contract
  /// @param _from Address to send tokens from
  /// @param _to Address to send tokens to
  /// @param _value Number of tokens to send
  /// @return success True if completed, false otherwise
  function transferFrom(TokenStorage storage self,
                        address _from,
                        address _to,
                        uint256 _value)
                        returns (bool success) {
    var _allowance = self.allowed[_from][msg.sender];
    bool err;
    uint256 balanceOwner;
    uint256 balanceSpender;

    (err,balanceOwner) = self.balances[_from].minus(_value);
    if(err) {
      ErrorMsg("Balance too low for transfer");
      return false;
    }

    (err,balanceSpender) = _allowance.minus(_value);
    if(err) {
      ErrorMsg("Transfer exceeds allowance");
      return false;
    }
    self.balances[_from] = balanceOwner;
    self.allowed[_from][msg.sender] = balanceSpender;
    self.balances[_to] = self.balances[_to] + _value;

    Transfer(_from, _to, _value);
    return true;
  }

  /// @dev Retrieve token balance for an account
  /// @param self Stored token from token contract
  /// @param _owner Address to retrieve balance of
  /// @return balance The number of tokens in the subject account
  function balanceOf(TokenStorage storage self, address _owner) constant returns (uint256 balance) {
    return self.balances[_owner];
  }

  /// @dev Authorize an account to send tokens on caller's behalf
  /// @param self Stored token from token contract
  /// @param _spender Address to authorize
  /// @param _value Number of tokens authorized account may send
  /// @return success True if completed, false otherwise
  function approve(TokenStorage storage self, address _spender, uint256 _value) returns (bool success) {
    self.allowed[msg.sender][_spender] = _value;
    Approval(msg.sender, _spender, _value);
    return true;
  }

  /// @dev Remaining tokens third party spender has to send
  /// @param self Stored token from token contract
  /// @param _owner Address of token holder
  /// @param _spender Address of authorized spender
  /// @return remaining Number of tokens spender has left in owner's account
  function allowance(TokenStorage storage self, address _owner, address _spender) constant returns (uint256 remaining) {
    return self.allowed[_owner][_spender];
  }
}

library BasicMathLib {
  event Err(string typeErr);

  /// @dev Multiplies two numbers and checks for overflow before returning.
  /// Does not throw but rather logs an Err event if there is overflow.
  /// @param a First number
  /// @param b Second number
  /// @return err False normally, or true if there is overflow
  /// @return res The product of a and b, or 0 if there is overflow
  function times(uint256 a, uint256 b) constant returns (bool err,uint256 res) {
    assembly{
      res := mul(a,b)
      jumpi(allGood, or(iszero(b), eq(div(res,b), a)))
      err := 1
      res := 0
      allGood:
    }
    if (err)
      Err("times func overflow");
  }

  /// @dev Divides two numbers but checks for 0 in the divisor first.
  /// Does not throw but rather logs an Err event if 0 is in the divisor.
  /// @param a First number
  /// @param b Second number
  /// @return err False normally, or true if `b` is 0
  /// @return res The quotient of a and b, or 0 if `b` is 0
  function dividedBy(uint256 a, uint256 b) constant returns (bool err,uint256 res) {
    assembly{
      jumpi(e, iszero(b))
      res := div(a,b)
      mstore(add(mload(0x40),0x20),res)
      return(mload(0x40),0x40)
      e:
    }
    Err("tried to divide by zero");
    return (true, 0);
  }

  /// @dev Adds two numbers and checks for overflow before returning.
  /// Does not throw but rather logs an Err event if there is overflow.
  /// @param a First number
  /// @param b Second number
  /// @return err False normally, or true if there is overflow
  /// @return res The sum of a and b, or 0 if there is overflow
  function plus(uint256 a, uint256 b) constant returns (bool err, uint256 res) {
    assembly{
      res := add(a,b)
      jumpi(allGood, and(eq(sub(res,b), a), gt(res,b)))
      err := 1
      res := 0
      allGood:
    }
    if (err)
      Err("plus func overflow");
  }

  /// @dev Subtracts two numbers and checks for underflow before returning.
  /// Does not throw but rather logs an Err event if there is underflow.
  /// @param a First number
  /// @param b Second number
  /// @return err False normally, or true if there is underflow
  /// @return res The difference between a and b, or 0 if there is underflow
  function minus(uint256 a, uint256 b) constant returns (bool err,uint256 res) {
    assembly{
      res := sub(a,b)
      jumpi(allGood, eq(and(eq(add(res,b), a), or(lt(res,a), eq(res,a))), 1))
      err := 1
      res := 0
      allGood:
    }
    if (err)
      Err("minus func underflow");
  }
}
