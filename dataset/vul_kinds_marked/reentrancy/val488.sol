pragma solidity ^0.4.13;

library SafeMath {

  /**
  * @dev Multiplies two numbers, throws on overflow.
  */
  function mul(uint256 _a, uint256 _b) internal pure returns (uint256 c) {
    // Gas optimization: this is cheaper than asserting 'a' not being zero, but the
    // benefit is lost if 'b' is also tested.
    // See: https://github.com/OpenZeppelin/openzeppelin-solidity/pull/522
    if (_a == 0) {
      return 0;
    }

    c = _a * _b;
    assert(c / _a == _b);
    return c;
  }

  /**
  * @dev Integer division of two numbers, truncating the quotient.
  */
  function div(uint256 _a, uint256 _b) internal pure returns (uint256) {
    // assert(_b > 0); // Solidity automatically throws when dividing by 0
    // uint256 c = _a / _b;
    // assert(_a == _b * c + _a % _b); // There is no case in which this doesn't hold
    return _a / _b;
  }

  /**
  * @dev Subtracts two numbers, throws on overflow (i.e. if subtrahend is greater than minuend).
  */
  function sub(uint256 _a, uint256 _b) internal pure returns (uint256) {
    assert(_b <= _a);
    return _a - _b;
  }

  /**
  * @dev Adds two numbers, throws on overflow.
  */
  function add(uint256 _a, uint256 _b) internal pure returns (uint256 c) {
    c = _a + _b;
    assert(c >= _a);
    return c;
  }
}

contract Ownable {
  address public owner;


  event OwnershipRenounced(address indexed previousOwner);
  event OwnershipTransferred(
    address indexed previousOwner,
    address indexed newOwner
  );


  /**
   * @dev The Ownable constructor sets the original `owner` of the contract to the sender
   * account.
   */
  constructor() public {
    owner = msg.sender;
  }

  /**
   * @dev Throws if called by any account other than the owner.
   */
  modifier onlyOwner() {
    require(msg.sender == owner);
    _;
  }

  /**
   * @dev Allows the current owner to relinquish control of the contract.
   * @notice Renouncing to ownership will leave the contract without an owner.
   * It will not be possible to call the functions with the `onlyOwner`
   * modifier anymore.
   */
  function renounceOwnership() public onlyOwner {
    emit OwnershipRenounced(owner);
    owner = address(0);
  }

  /**
   * @dev Allows the current owner to transfer control of the contract to a newOwner.
   * @param _newOwner The address to transfer ownership to.
   */
  function transferOwnership(address _newOwner) public onlyOwner {
    _transferOwnership(_newOwner);
  }

  /**
   * @dev Transfers control of the contract to a newOwner.
   * @param _newOwner The address to transfer ownership to.
   */
  function _transferOwnership(address _newOwner) internal {
    require(_newOwner != address(0));
    emit OwnershipTransferred(owner, _newOwner);
    owner = _newOwner;
  }
}

contract Pausable is Ownable {
  event Pause();
  event Unpause();

  bool public paused = false;


  /**
   * @dev Modifier to make a function callable only when the contract is not paused.
   */
  modifier whenNotPaused() {
    require(!paused);
    _;
  }

  /**
   * @dev Modifier to make a function callable only when the contract is paused.
   */
  modifier whenPaused() {
    require(paused);
    _;
  }

  /**
   * @dev called by the owner to pause, triggers stopped state
   */
  function pause() public onlyOwner whenNotPaused {
    paused = true;
    emit Pause();
  }

  /**
   * @dev called by the owner to unpause, returns to normal state
   */
  function unpause() public onlyOwner whenPaused {
    paused = false;
    emit Unpause();
  }
}

contract CanReclaimToken is Ownable {
  using SafeERC20 for ERC20Basic;

  /**
   * @dev Reclaim all ERC20Basic compatible tokens
   * @param _token ERC20Basic The address of the token contract
   */
  function reclaimToken(ERC20Basic _token) external onlyOwner {
    uint256 balance = _token.balanceOf(this);
    _token.safeTransfer(owner, balance);
  }

}

contract ERC20Basic {
  function totalSupply() public view returns (uint256);
  function balanceOf(address _who) public view returns (uint256);
  function transfer(address _to, uint256 _value) public returns (bool);
  event Transfer(address indexed from, address indexed to, uint256 value);
}

contract BasicToken is ERC20Basic {
  using SafeMath for uint256;

  mapping(address => uint256) internal balances;

  uint256 internal totalSupply_;

  /**
  * @dev Total number of tokens in existence
  */
  function totalSupply() public view returns (uint256) {
    return totalSupply_;
  }

  /**
  * @dev Transfer token for a specified address
  * @param _to The address to transfer to.
  * @param _value The amount to be transferred.
  */
  function transfer(address _to, uint256 _value) public returns (bool) {
    require(_value <= balances[msg.sender]);
    require(_to != address(0));

    balances[msg.sender] = balances[msg.sender].sub(_value);
    balances[_to] = balances[_to].add(_value);
    emit Transfer(msg.sender, _to, _value);
    return true;
  }

  /**
  * @dev Gets the balance of the specified address.
  * @param _owner The address to query the the balance of.
  * @return An uint256 representing the amount owned by the passed address.
  */
  function balanceOf(address _owner) public view returns (uint256) {
    return balances[_owner];
  }

}

contract BurnableToken is BasicToken {

  event Burn(address indexed burner, uint256 value);

  /**
   * @dev Burns a specific amount of tokens.
   * @param _value The amount of token to be burned.
   */
  function burn(uint256 _value) public {
    _burn(msg.sender, _value);
  }

  function _burn(address _who, uint256 _value) internal {
    require(_value <= balances[_who]);
    // no need to require value <= totalSupply, since that would imply the
    // sender's balance is greater than the totalSupply, which *should* be an assertion failure

    balances[_who] = balances[_who].sub(_value);
    totalSupply_ = totalSupply_.sub(_value);
    emit Burn(_who, _value);
    emit Transfer(_who, address(0), _value);
  }
}

contract ERC20 is ERC20Basic {
  function allowance(address _owner, address _spender)
    public view returns (uint256);

  function transferFrom(address _from, address _to, uint256 _value)
    public returns (bool);

  function approve(address _spender, uint256 _value) public returns (bool);
  event Approval(
    address indexed owner,
    address indexed spender,
    uint256 value
  );
}

library SafeERC20 {
  function safeTransfer(
    ERC20Basic _token,
    address _to,
    uint256 _value
  )
    internal
  {
    require(_token.transfer(_to, _value));
  }

  function safeTransferFrom(
    ERC20 _token,
    address _from,
    address _to,
    uint256 _value
  )
    internal
  {
    require(_token.transferFrom(_from, _to, _value));
  }

  function safeApprove(
    ERC20 _token,
    address _spender,
    uint256 _value
  )
    internal
  {
    require(_token.approve(_spender, _value));
  }
}

contract StandardToken is ERC20, BasicToken {

  mapping (address => mapping (address => uint256)) internal allowed;


  /**
   * @dev Transfer tokens from one address to another
   * @param _from address The address which you want to send tokens from
   * @param _to address The address which you want to transfer to
   * @param _value uint256 the amount of tokens to be transferred
   */
  function transferFrom(
    address _from,
    address _to,
    uint256 _value
  )
    public
    returns (bool)
  {
    require(_value <= balances[_from]);
    require(_value <= allowed[_from][msg.sender]);
    require(_to != address(0));

    balances[_from] = balances[_from].sub(_value);
    balances[_to] = balances[_to].add(_value);
    allowed[_from][msg.sender] = allowed[_from][msg.sender].sub(_value);
    emit Transfer(_from, _to, _value);
    return true;
  }

  /**
   * @dev Approve the passed address to spend the specified amount of tokens on behalf of msg.sender.
   * Beware that changing an allowance with this method brings the risk that someone may use both the old
   * and the new allowance by unfortunate transaction ordering. One possible solution to mitigate this
   * race condition is to first reduce the spender's allowance to 0 and set the desired value afterwards:
   * https://github.com/ethereum/EIPs/issues/20#issuecomment-263524729
   * @param _spender The address which will spend the funds.
   * @param _value The amount of tokens to be spent.
   */
  function approve(address _spender, uint256 _value) public returns (bool) {
    allowed[msg.sender][_spender] = _value;
    emit Approval(msg.sender, _spender, _value);
    return true;
  }

  /**
   * @dev Function to check the amount of tokens that an owner allowed to a spender.
   * @param _owner address The address which owns the funds.
   * @param _spender address The address which will spend the funds.
   * @return A uint256 specifying the amount of tokens still available for the spender.
   */
  function allowance(
    address _owner,
    address _spender
   )
    public
    view
    returns (uint256)
  {
    return allowed[_owner][_spender];
  }

  /**
   * @dev Increase the amount of tokens that an owner allowed to a spender.
   * approve should be called when allowed[_spender] == 0. To increment
   * allowed value is better to use this function to avoid 2 calls (and wait until
   * the first transaction is mined)
   * From MonolithDAO Token.sol
   * @param _spender The address which will spend the funds.
   * @param _addedValue The amount of tokens to increase the allowance by.
   */
  function increaseApproval(
    address _spender,
    uint256 _addedValue
  )
    public
    returns (bool)
  {
    allowed[msg.sender][_spender] = (
      allowed[msg.sender][_spender].add(_addedValue));
    emit Approval(msg.sender, _spender, allowed[msg.sender][_spender]);
    return true;
  }

  /**
   * @dev Decrease the amount of tokens that an owner allowed to a spender.
   * approve should be called when allowed[_spender] == 0. To decrement
   * allowed value is better to use this function to avoid 2 calls (and wait until
   * the first transaction is mined)
   * From MonolithDAO Token.sol
   * @param _spender The address which will spend the funds.
   * @param _subtractedValue The amount of tokens to decrease the allowance by.
   */
  function decreaseApproval(
    address _spender,
    uint256 _subtractedValue
  )
    public
    returns (bool)
  {
    uint256 oldValue = allowed[msg.sender][_spender];
    if (_subtractedValue >= oldValue) {
      allowed[msg.sender][_spender] = 0;
    } else {
      allowed[msg.sender][_spender] = oldValue.sub(_subtractedValue);
    }
    emit Approval(msg.sender, _spender, allowed[msg.sender][_spender]);
    return true;
  }

}

contract PausableToken is StandardToken, Pausable {

  function transfer(
    address _to,
    uint256 _value
  )
    public
    whenNotPaused
    returns (bool)
  {
    return super.transfer(_to, _value);
  }

  function transferFrom(
    address _from,
    address _to,
    uint256 _value
  )
    public
    whenNotPaused
    returns (bool)
  {
    return super.transferFrom(_from, _to, _value);
  }

  function approve(
    address _spender,
    uint256 _value
  )
    public
    whenNotPaused
    returns (bool)
  {
    return super.approve(_spender, _value);
  }

  function increaseApproval(
    address _spender,
    uint _addedValue
  )
    public
    whenNotPaused
    returns (bool success)
  {
    return super.increaseApproval(_spender, _addedValue);
  }

  function decreaseApproval(
    address _spender,
    uint _subtractedValue
  )
    public
    whenNotPaused
    returns (bool success)
  {
    return super.decreaseApproval(_spender, _subtractedValue);
  }
}

contract MenloSaleBase is Ownable {
  using SafeMath for uint256;

  // Whitelisted investors
  mapping (address => bool) public whitelist;

  // Special role used exclusively for managing the whitelist
  address public whitelister;

  // manual early close flag
  bool public isFinalized;

  // cap for crowdsale in wei
  uint256 public cap;

  // The token being sold
  MenloToken public token;

  // start and end timestamps where contributions are allowed (both inclusive)
  uint256 public startTime;
  uint256 public endTime;

  // address where funds are collected
  address public wallet;

  // amount of raised money in wei
  uint256 public weiRaised;

  /**
   * @dev Throws if called by any account other than the whitelister.
   */
  modifier onlyWhitelister() {
    require(msg.sender == whitelister, "Sender should be whitelister");
    _;
  }

  /**
   * event for token purchase logging
   * @param purchaser who bought the tokens
   * @param value weis paid for purchase
   * @param amount amount of tokens purchased
   */
  event TokenPurchase(address indexed purchaser, uint256 value, uint256 amount);

  /**
   * event for token redemption logging
   * @param purchaser who bought the tokens
   * @param amount amount of tokens redeemed
   */
  event TokenRedeem(address indexed purchaser, uint256 amount);

  // termination early or otherwise
  event Finalized();

  event TokensRefund(uint256 amount);

  /**
   * event refund of excess ETH if purchase is above the cap
   * @param amount amount of ETH (in wei) refunded
   */
  event Refund(address indexed purchaser, uint256 amount);

  constructor(
      MenloToken _token,
      uint256 _startTime,
      uint256 _endTime,
      uint256 _cap,
      address _wallet
  ) public {
    require(_startTime >= getBlockTimestamp(), "Start time should be in the future");
    require(_endTime >= _startTime, "End time should be after start time");
    require(_wallet != address(0), "Wallet address should be non-zero");
    require(_token != address(0), "Token address should be non-zero");
    require(_cap > 0, "Cap should be greater than zero");

    token = _token;

    startTime = _startTime;
    endTime = _endTime;
    cap = _cap;
    wallet = _wallet;
  }

  // fallback function can be used to buy tokens
  function () public payable {
    buyTokens();
  }

  // Abstract methods
  function calculateBonusRate() public view returns (uint256);
  function buyTokensHook(uint256 _tokens) internal;

  function buyTokens() public payable returns (uint256) {
    require(whitelist[msg.sender], "Expected msg.sender to be whitelisted");
    checkFinalize();
    require(!isFinalized, "Should not be finalized when purchasing");
    require(getBlockTimestamp() >= startTime && getBlockTimestamp() <= endTime, "Should be during sale");
    require(msg.value != 0, "Value should not be zero");
    require(token.balanceOf(this) > 0, "This contract must have tokens");

    uint256 _weiAmount = msg.value;

    uint256 _remainingToFund = cap.sub(weiRaised);
    if (_weiAmount > _remainingToFund) {
      _weiAmount = _remainingToFund;
    }

    uint256 _totalTokens = _weiAmount.mul(calculateBonusRate());
    if (_totalTokens > token.balanceOf(this)) {
      // Change _wei to buy rest of remaining tokens
      _weiAmount = token.balanceOf(this).div(calculateBonusRate());
    }

    token.unpause();
    weiRaised = weiRaised.add(_weiAmount);

    forwardFunds(_weiAmount);
    uint256 _weiToReturn = msg.value.sub(_weiAmount);
    if (_weiToReturn > 0) {
      msg.sender.transfer(_weiToReturn);
      emit Refund(msg.sender, _weiToReturn);
    }

    uint256 _tokens = ethToTokens(_weiAmount);
    emit TokenPurchase(msg.sender, _weiAmount, _tokens);
    buyTokensHook(_tokens);
    token.pause();

    checkFinalize();

    return _tokens;
  }

  // Allows the owner to take back the tokens that are assigned to the sale contract.
  function refund() external onlyOwner returns (bool) {
    require(hasEnded(), "Sale should have ended when refunding");
    uint256 _tokens = token.balanceOf(address(this));

    if (_tokens == 0) {
      return false;
    }

    require(token.transfer(owner, _tokens), "Expected token transfer to succeed");

    emit TokensRefund(_tokens);

    return true;
  }

  /// @notice interface for founders to whitelist investors
  /// @param _addresses array of investors
  /// @param _status enable or disable
  function whitelistAddresses(address[] _addresses, bool _status) public onlyWhitelister {
    for (uint256 i = 0; i < _addresses.length; i++) {
      address _investorAddress = _addresses[i];
      if (whitelist[_investorAddress] != _status) {
        whitelist[_investorAddress] = _status;
      }
    }
  }

  function setWhitelister(address _whitelister) public onlyOwner {
    whitelister = _whitelister;
  }

  function checkFinalize() public {
    if (hasEnded()) {
      finalize();
    }
  }

  function emergencyFinalize() public onlyOwner {
    finalize();
  }

  function withdraw() public onlyOwner {
    owner.transfer(address(this).balance);
  }

  function hasEnded() public constant returns (bool) {
    if (isFinalized) {
      return true;
    }
    bool _capReached = weiRaised >= cap;
    bool _passedEndTime = getBlockTimestamp() > endTime;
    return _passedEndTime || _capReached;
  }

  // @dev does not require that crowdsale `hasEnded()` to leave safegaurd
  // in place if ETH rises in price too much during crowdsale.
  // Allows team to close early if cap is exceeded in USD in this event.
  function finalize() internal {
    require(!isFinalized, "Should not be finalized when finalizing");
    emit Finalized();
    isFinalized = true;
    token.transferOwnership(owner);
  }

  // send ether to the fund collection wallet
  // override to create custom fund forwarding mechanisms
  function forwardFunds(uint256 _amount) internal {
    wallet.transfer(_amount);
  }

  function ethToTokens(uint256 _ethAmount) internal view returns (uint256) {
    return _ethAmount.mul(calculateBonusRate());
  }

  function getBlockTimestamp() internal view returns (uint256) {
    return block.timestamp;
  }
}

contract MenloToken is PausableToken, BurnableToken, CanReclaimToken {

  // Token properties
  string public constant name = 'Menlo One';
  string public constant symbol = 'ONE';

  uint8 public constant decimals = 18;
  uint256 private constant token_factor = 10**uint256(decimals);

  // 1 billion ONE tokens in units divisible up to 18 decimals
  uint256 public constant INITIAL_SUPPLY    = 1000000000 * token_factor;

  uint256 public constant PUBLICSALE_SUPPLY = 354000000 * token_factor;
  uint256 public constant GROWTH_SUPPLY     = 246000000 * token_factor;
  uint256 public constant TEAM_SUPPLY       = 200000000 * token_factor;
  uint256 public constant ADVISOR_SUPPLY    = 100000000 * token_factor;
  uint256 public constant PARTNER_SUPPLY    = 100000000 * token_factor;

  /**
   * @dev Magic value to be returned upon successful reception of Menlo Tokens
   */
  bytes4 internal constant ONE_RECEIVED = 0x150b7a03;

  address public crowdsale;
  address public teamTimelock;
  address public advisorTimelock;

  modifier notInitialized(address saleAddress) {
    require(address(saleAddress) == address(0), "Expected address to be null");
    _;
  }

  constructor(address _growth, address _teamTimelock, address _advisorTimelock, address _partner) public {
    assert(INITIAL_SUPPLY > 0);
    assert((PUBLICSALE_SUPPLY + GROWTH_SUPPLY + TEAM_SUPPLY + ADVISOR_SUPPLY + PARTNER_SUPPLY) == INITIAL_SUPPLY);

    uint256 _poolTotal = GROWTH_SUPPLY + TEAM_SUPPLY + ADVISOR_SUPPLY + PARTNER_SUPPLY;
    uint256 _availableForSales = INITIAL_SUPPLY - _poolTotal;

    assert(_availableForSales == PUBLICSALE_SUPPLY);

    teamTimelock = _teamTimelock;
    advisorTimelock = _advisorTimelock;

    mint(msg.sender, _availableForSales);
    mint(_growth, GROWTH_SUPPLY);
    mint(_teamTimelock, TEAM_SUPPLY);
    mint(_advisorTimelock, ADVISOR_SUPPLY);
    mint(_partner, PARTNER_SUPPLY);

    assert(totalSupply_ == INITIAL_SUPPLY);
    pause();
  }

  function initializeCrowdsale(address _crowdsale) public onlyOwner notInitialized(crowdsale) {
    unpause();
    transfer(_crowdsale, balances[msg.sender]);  // Transfer left over balance after private presale allocations
    crowdsale = _crowdsale;
    pause();
    transferOwnership(_crowdsale);
  }

  function mint(address _to, uint256 _amount) internal {
    balances[_to] = _amount;
    totalSupply_ = totalSupply_.add(_amount);
    emit Transfer(address(0), _to, _amount);
  }

  /**
   * @dev Safely transfers the ownership of a given token ID to another address
   * If the target address is a contract, it must implement `onERC721Received`,
   * which is called upon a safe transfer, and return the magic value `bytes4(0x150b7a03)`;
   * otherwise, the transfer is reverted.
   * Requires the msg sender to be the owner, approved, or operator
   * @param _to address to receive the tokens.  Must be a MenloTokenReceiver based contract
   * @param _value uint256 number of tokens to transfer
   * @param _action uint256 action to perform in target _to contract
   * @param _data bytes data to send along with a safe transfer check
   **/
  function transferAndCall(address _to, uint256 _value, uint256 _action, bytes _data) public returns (bool) {
    if (transfer(_to, _value)) {
      require (MenloTokenReceiver(_to).onTokenReceived(msg.sender, _value, _action, _data) == ONE_RECEIVED, "Target contract onTokenReceived failed");
      return true;
    }

    return false;
  }
}

contract MenloTokenReceiver {

    /*
     * @dev Address of the MenloToken contract
     */
    MenloToken token;

    constructor(MenloToken _tokenContract) public {
        token = _tokenContract;
    }

    /**
     * @dev Magic value to be returned upon successful reception of Menlo Tokens
     */
    bytes4 internal constant ONE_RECEIVED = 0x150b7a03;

    /**
     * @dev Throws if called by any account other than the Menlo Token contract.
     */
    modifier onlyTokenContract() {
        require(msg.sender == address(token));
        _;
    }

    /**
     * @notice Handle the receipt of Menlo Tokens
     * @dev The MenloToken contract calls this function on the recipient
     * after a `transferAndCall`. This function MAY throw to revert and reject the
     * transfer. Return of other than the magic value MUST result in the
     * transaction being reverted.
     * Warning: this function must call the onlyTokenContract modifier to trust
     * the transfer took place
     * @param _from The address which previously owned the token
     * @param _value Number of tokens that were transfered
     * @param _action Used to define enumeration of possible functions to call
     * @param _data Additional data with no specified format
     * @return `bytes4(0x150b7a03)`
     */
    function onTokenReceived(
        address _from,
        uint256 _value,
        uint256 _action,
        bytes _data
    ) public /* onlyTokenContract */ returns(bytes4);
}

contract MenloTokenSale is MenloSaleBase {

  // Timestamps for the bonus periods, set in the constructor
  uint256 public HOUR1;
  uint256 public WEEK1;
  uint256 public WEEK2;
  uint256 public WEEK3;
  uint256 public WEEK4;

  constructor(
    MenloToken _token,
    uint256 _startTime,
    uint256 _endTime,
    uint256 _cap,
    address _wallet
  ) MenloSaleBase(
    _token,
    _startTime,
    _endTime,
    _cap,
    _wallet
  ) public {
    HOUR1 = startTime + 1 hours;
    WEEK1 = startTime + 1 weeks;
    WEEK2 = startTime + 2 weeks;
    WEEK3 = startTime + 3 weeks;
  }

  // Hour 1: 30% Bonus
  // Week 1: 15% Bonus
  // Week 2: 10% Bonus
  // Week 3: 5% Bonus
  // Week 4: 0% Bonus
  function calculateBonusRate() public view returns (uint256) {
    uint256 _bonusRate = 12000;

    uint256 _currentTime = getBlockTimestamp();
    if (_currentTime > startTime && _currentTime <= HOUR1) {
      _bonusRate =  15600;
    } else if (_currentTime <= WEEK1) {
      _bonusRate =  13800; // week 1
    } else if (_currentTime <= WEEK2) {
      _bonusRate =  13200; // week 2
    } else if (_currentTime <= WEEK3) {
      _bonusRate =  12600; // week 3
    }
    return _bonusRate;
  }

  function buyTokensHook(uint256 _tokens) internal {
    token.transfer(msg.sender, _tokens);
    emit TokenRedeem(msg.sender, _tokens);
  }
}
