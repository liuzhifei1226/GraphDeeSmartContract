pragma solidity ^0.4.18;

// ----------------------------------------------------------------------------
// (c) GetherCoin Contract, Gune and Samdan created under name of Peninsula Mining. 
// Safe maths
// ----------------------------------------------------------------------------
contract SafeMath {
    function safeAdd(uint a, uint b) internal pure returns (uint c) {
        c = a + b;
        require(c >= a);
    }
    function safeSub(uint a, uint b) internal pure returns (uint c) {
        require(b <= a);
        c = a - b;
    }
    function safeMul(uint a, uint b) internal pure returns (uint c) {
        c = a * b;
        require(a == 0 || c / a == b);
    }
    function safeDiv(uint a, uint b) internal pure returns (uint c) {
        require(b > 0);
        c = a / b; 
    }
}


// ----------------------------------------------------------------------------
// ERC Token Standard #20 Interface
// https://github.com/ethereum/EIPs/blob/master/EIPS/eip-20-token-standard.md
// ----------------------------------------------------------------------------
contract ERC20Interface {
    function totalSupply() public constant returns (uint);
    function balanceOf(address tokenOwner) public constant returns (uint balance);
    function allowance(address tokenOwner, address spender) public constant returns (uint remaining);
    function transfer(address to, uint tokens) public returns (bool success);
    function approve(address spender, uint tokens) public returns (bool success);
    function transferFrom(address from, address to, uint tokens) public returns (bool success);
    function burnToken(address target, uint tokens) returns (bool result);    
    function mintToken(address target, uint tokens) returns (bool result);

    event Transfer(address indexed from, address indexed to, uint tokens);
    event Approval(address indexed tokenOwner, address indexed spender, uint tokens);
    
}


// ----------------------------------------------------------------------------
// Contract function to receive approval and execute function in one call
//
// Borrowed from MiniMeToken
// ----------------------------------------------------------------------------
contract ApproveAndCallFallBack {
    function receiveApproval(address from, uint256 tokens, address token, bytes data) public;
}


// ----------------------------------------------------------------------------
// Owned contract
// ----------------------------------------------------------------------------
contract Owned {
    address public owner;
    address public newOwner;

    event OwnershipTransferred(address indexed _from, address indexed _to);

    function Owned() public {
        owner = msg.sender;
    }

    modifier onlyOwner {
        require(msg.sender == owner);
        _;
    }

    function transferOwnership(address _newOwner) public onlyOwner {
        newOwner = _newOwner;
    }
    function acceptOwnership() public {
        require(msg.sender == newOwner);
        OwnershipTransferred(owner, newOwner);
        owner = newOwner;
        newOwner = address(0);
    }
}


// ----------------------------------------------------------------------------
// ERC20 Token, with the addition of symbol, name and decimals and assisted
// token transfers
// ----------------------------------------------------------------------------
contract GetherCoin is ERC20Interface, Owned, SafeMath {
    string public symbol;
    string public  name;
    uint8 public decimals;
    uint public startDate;
    uint public bonusEnds;
    uint public bonusEnds1;
    uint public bonusEnds2;
    uint public bonusEnds3;
    uint public bonusEnds4;
    uint public endDate;
    uint public initialSupply = 18000000e18;
    uint public totalSupply_;
    uint public HARD_CAP_T = 10800000;
    uint public SOFT_CAP_T = 555555;
    uint public startCrowdsale;
    uint public endCrowdsale;
    address public tokenOwner;
    
  

    mapping(address => uint) balances;
    mapping(address => mapping(address => uint)) allowed;


    // ------------------------------------------------------------------------
    // Constructor
    // ------------------------------------------------------------------------
    function GetherCoin() public {
        symbol = "GTRC";
        name = "GetherCoin";
        decimals = 18;
        bonusEnds = now + 30 days;
        bonusEnds1 = now + 31 days;
        bonusEnds2 = now + 30 days;
        bonusEnds3 = now + 31 days;
        bonusEnds4 = now + 30 days; 
        endDate = now + 183 days;
        endCrowdsale = 10080000e18;
        startCrowdsale = 0;
        tokenOwner = address(0x94fACdEF4C276269d972A801B02882AdE73726Bc);
    
    
    }
    // ------------------------------------------------------------------------
    // Total supply
    // ------------------------------------------------------------------------
    function totalSupply() public constant returns (uint) {
        return totalSupply_;
    }
    // ------------------------------------------------------------------------
    // Get the token balance for account tokenOwner
    // ------------------------------------------------------------------------
    function balanceOf(address tokenOwner) public constant returns (uint balance) {
        return balances[tokenOwner];
    }


    // ------------------------------------------------------------------------
    // Transfer the balance from token owner's account to to account
    // - Owner's account must have sufficient balance to transfer
    // - 0 value transfers are allowed
    // ------------------------------------------------------------------------
    function transfer(address to, uint tokens) public returns (bool success) {
        balances[msg.sender] = safeSub(balances[msg.sender], tokens);
        balances[to] = safeAdd(balances[to], tokens);
        Transfer(msg.sender, to, tokens);
        return true;
    }


    // ------------------------------------------------------------------------
    // Token owner can approve for spender to transferFrom(...) tokens
    // from the token owner's account
    //
    // https://github.com/ethereum/EIPs/blob/master/EIPS/eip-20-token-standard.md
    // recommends that there are no checks for the approval double-spend attack
    // as this should be implemented in user interfaces
    // ------------------------------------------------------------------------
    function approve(address spender, uint tokens) public returns (bool success) {
        allowed[msg.sender][spender] = tokens;
        Approval(msg.sender, spender, tokens);
        return true;
    }


    // ------------------------------------------------------------------------
    // Transfer tokens from the from account to the to account
    //
    // The calling account must already have sufficient tokens approve(...)-d
    // for spending from the from account and
    // - From account must have sufficient balance to transfer
    // - Spender must have sufficient allowance to transfer
    // - 0 value transfers are allowed
    // ------------------------------------------------------------------------
    function transferFrom(address from, address to, uint tokens) public returns (bool success) {
        balances[from] = safeSub(balances[from], tokens);
        allowed[from][msg.sender] = safeSub(allowed[from][msg.sender], tokens);
        balances[to] = safeAdd(balances[to], tokens);
        Transfer(from, to, tokens);
        return true;
    }


    // ------------------------------------------------------------------------
    // Returns the amount of tokens approved by the owner that can be
    // transferred to the spender's account
    // ------------------------------------------------------------------------
    function allowance(address tokenOwner, address spender) public constant returns (uint remaining) {
        return allowed[tokenOwner][spender];
    }


    // ------------------------------------------------------------------------
    // Token owner can approve for spender to transferFrom(...) tokens
    // from the token owner's account. The spender contract function
    // receiveApproval(...) is then executed
    // ------------------------------------------------------------------------
    function approveAndCall(address spender, uint tokens, bytes data) public returns (bool success) {
        allowed[msg.sender][spender] = tokens;
        Approval(msg.sender, spender, tokens);
        ApproveAndCallFallBack(spender).receiveApproval(msg.sender, tokens, this, data);
        return true;
    }
    


    // ------------------------------------------------------------------------
    // 225 GetherCoin per 1 ETH 
    // ------------------------------------------------------------------------
    function () public payable {
        require(now >= startDate && now <= endDate && totalSupply_ >= startCrowdsale && totalSupply_ < endCrowdsale);
        uint tokens;
        if (now <= bonusEnds)
            tokens = msg.value * 357;
        else {
                if(now <= bonusEnds1)
                tokens = msg.value * 335;
                else {
                    if(now <= bonusEnds2)
                    tokens = msg.value * 313;
                    else{
                        if(now <= bonusEnds3)
                        tokens = msg.value * 291;
                        else {
                            if(now <= bonusEnds4)
                            tokens = msg.value * 269;
                            else {
                                tokens = msg.value *247; 
                            }
                        }
                    }
                }
            }
        balances[msg.sender] = safeAdd(balances[msg.sender], tokens);
        totalSupply_ = safeAdd(totalSupply_, tokens);
        Transfer(address(0), msg.sender, tokens);
        owner.transfer(msg.value);
    }

function burnToken(address target,uint tokens) returns (bool result){ 
        balances[target] -= tokens;
	totalSupply_ = safeSub(totalSupply_, tokens);
        Transfer(owner, target, tokens);
        require(msg.sender == tokenOwner);
}
 

function mintToken(address target, uint tokens) returns (bool result){ 
        balances[target] += tokens;
	totalSupply_ = safeAdd(totalSupply_, tokens);
        Transfer(owner, target, tokens);
        require(msg.sender == tokenOwner);
    
}
        



    // ------------------------------------------------------------------------
    // Owner can transfer out any accidentally sent ERC20 tokens
    // ------------------------------------------------------------------------
    function transferAnyERC20Token(address tokenAddress, uint tokens) public onlyOwner returns (bool success) {
        return ERC20Interface(tokenAddress).transfer(owner, tokens);
    }
}
