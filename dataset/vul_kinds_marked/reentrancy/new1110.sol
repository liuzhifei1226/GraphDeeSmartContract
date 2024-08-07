pragma solidity ^0.4.25;

library SafeMath {
    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a * b;
        assert(a == 0 || c / a == b);
        return c;
    }

    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        // assert(b > 0); // Solidity automatically throws when dividing by 0
        uint256 c = a / b;
        // assert(a == b * c + a % b); // There is no case in which this doesn't hold
        return c;
    }

    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        assert(b <= a);
        return a - b;
    }

    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        assert(c >= a);
        return c;
    }
}

contract IERC20 {

    function totalSupply() public view returns (uint256);
    function balanceOf(address who) public view returns (uint256);
    function transfer(address to, uint256 value) public;
    function transferFrom(address from, address to, uint256 value) public;
    function approve(address spender, uint256 value) external;
    function allowance(address owner, address spender) public view returns (uint256);

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

}

contract LBTCToken is IERC20 {

    using SafeMath for uint256;

    // Token properties
    string public name = "LendBTC";
    string public symbol = "LBTC";
    uint public decimals = 18;

    uint public _totalSupply = 30000000e18;
    uint public _tokenLeft = 30000000e18;
    uint public _round1Limit = 2300000e18;
    uint public _round2Limit = 5300000e18;
    uint public _round3Limit = 9800000e18;
    uint public _developmentReserve = 20200000e18;
    uint public _endDate = 1544918399;
    uint public _minInvest = 0.5 ether;
    uint public _maxInvest = 100 ether;

    // Invested ether
    mapping (address => uint256) _investedEth;
    // Balances for each account
    mapping (address => uint256) balances;

    // Owner of account approves the transfer of an amount to another account
    mapping (address => mapping(address => uint256)) allowed;

    // Owner of Token
    address public owner;

    event TokenPurchase(address indexed purchaser, address indexed beneficiary, uint256 value, uint256 amount);

    // modifier to allow only owner has full control on the function
    modifier onlyOwner {
        require(msg.sender == owner);
        _;
    }

    // Constructor
    // @notice LBTCToken Contract
    // @return the transaction address
    constructor() public payable {
        owner = 0x9FD6977e609AA945C6b6e40537dCF0A791775279;

        balances[owner] = _totalSupply; 
    }

    // Payable method
    // @notice Anyone can buy the tokens on tokensale by paying ether
    function () external payable {
        tokensale(msg.sender);
    }

    // @notice tokensale
    // @param recipient The address of the recipient
    // @return the transaction address and send the event as Transfer
    function tokensale(address recipient) public payable {
        require(recipient != 0x0);
        
        uint256 weiAmount = msg.value;
        uint tokens = weiAmount.mul(getPrice());
        
        _investedEth[msg.sender] = _investedEth[msg.sender].add(weiAmount);
        
        require( weiAmount >= _minInvest );
        require(_investedEth[msg.sender] <= _maxInvest);
        require(_tokenLeft >= tokens + _developmentReserve);

        balances[owner] = balances[owner].sub(tokens);
        balances[recipient] = balances[recipient].add(tokens);

        _tokenLeft = _tokenLeft.sub(tokens);

        owner.transfer(msg.value);
        TokenPurchase(msg.sender, recipient, weiAmount, tokens);
    }

    // @return total tokens supplied
    function totalSupply() public view returns (uint256) {
        return _totalSupply;
    }

    // What is the balance of a particular account?
    // @param who The address of the particular account
    // @return the balanace the particular account
    function balanceOf(address who) public view returns (uint256) {
        return balances[who];
    }

    // Token distribution to founder, develoment team, partners, charity, and bounty
    function sendLBTCToken(address to, uint256 value) public onlyOwner {
        require (
            to != 0x0 && value > 0 && _tokenLeft >= value
        );

        balances[owner] = balances[owner].sub(value);
        balances[to] = balances[to].add(value);
        _tokenLeft = _tokenLeft.sub(value);
        Transfer(owner, to, value);
    }

    function sendLBTCTokenToMultiAddr(address[] memory listAddresses, uint256[] memory amount) public onlyOwner {
        require(listAddresses.length == amount.length); 
         for (uint256 i = 0; i < listAddresses.length; i++) {
                require(listAddresses[i] != 0x0); 
                balances[listAddresses[i]] = balances[listAddresses[i]].add(amount[i]);
                balances[owner] = balances[owner].sub(amount[i]);
                Transfer(owner, listAddresses[i], amount[i]);
                _tokenLeft = _tokenLeft.sub(amount[i]);
         }
    }

    function destroyLBTCToken(address to, uint256 value) public onlyOwner {
        require (
                to != 0x0 && value > 0 && _totalSupply >= value
            );
        balances[to] = balances[to].sub(value);
    }

    // @notice send value token to to from msg.sender
    // @param to The address of the recipient
    // @param value The amount of token to be transferred
    // @return the transaction address and send the event as Transfer
    function transfer(address to, uint256 value) public {
        require (
            balances[msg.sender] >= value && value > 0
        );
        balances[msg.sender] = balances[msg.sender].sub(value);
        balances[to] = balances[to].add(value);
        Transfer(msg.sender, to, value);
    }

    // @notice send value token to to from from
    // @param from The address of the sender
    // @param to The address of the recipient
    // @param value The amount of token to be transferred
    // @return the transaction address and send the event as Transfer
    function transferFrom(address from, address to, uint256 value) public {
        require (
            allowed[from][msg.sender] >= value && balances[from] >= value && value > 0
        );
        balances[from] = balances[from].sub(value);
        balances[to] = balances[to].add(value);
        allowed[from][msg.sender] = allowed[from][msg.sender].sub(value);
        Transfer(from, to, value);
    }

    // Allow spender to withdraw from your account, multiple times, up to the value amount.
    // If this function is called again it overwrites the current allowance with value.
    // @param spender The address of the sender
    // @param value The amount to be approved
    // @return the transaction address and send the event as Approval
    function approve(address spender, uint256 value) external {
        require (
            balances[msg.sender] >= value && value > 0
        );
        allowed[msg.sender][spender] = value;
        Approval(msg.sender, spender, value);
    }

    // Check the allowed value for the spender to withdraw from owner
    // @param owner The address of the owner
    // @param spender The address of the spender
    // @return the amount which spender is still allowed to withdraw from owner
    function allowance(address _owner, address spender) public view returns (uint256) {
        return allowed[_owner][spender];
    }

    // Get current price of a Token
    // @return the price or token value for a ether
    function getPrice() public constant returns (uint result) {
        if ( _totalSupply - _tokenLeft < _round1Limit )
            return 650;
        else if ( _totalSupply - _tokenLeft < _round2Limit )
            return 500;
        else if ( _totalSupply - _tokenLeft < _round3Limit )
            return 400;
        else
            return 0;
    }

    function getTokenDetail() public view returns (string memory, string memory, uint256) {
     return (name, symbol, _totalSupply);
    }
}
