{"ERC20Interface.sol":{"content":"pragma solidity ^0.5.0;\n\n// ----------------------------------------------------------------------------\n// ERC Token Standard #20 Interface\n// https://github.com/ethereum/EIPs/blob/master/EIPS/eip-20-token-standard.md\n// ----------------------------------------------------------------------------\ncontract ERC20Interface {\n    function totalSupply() public view returns (uint);\n    function balanceOf(address tokenOwner) public view returns (uint balance);\n    function allowance(address tokenOwner, address spender) public view returns (uint remaining);\n    function transfer(address to, uint tokens) public returns (bool success);\n    function approve(address spender, uint tokens) public returns (bool success);\n    function transferFrom(address from, address to, uint tokens) public returns (bool success);\n\n    event Transfer(address indexed from, address indexed to, uint tokens);\n    event Approval(address indexed tokenOwner, address indexed spender, uint tokens);\n}\n"},"RMBT.sol":{"content":"pragma solidity ^0.5.0;\n\nimport \"./SafeMath.sol\";\nimport \"./ERC20Interface.sol\";\n\n// ----------------------------------------------------------------------------\n// \u0027RMBT\u0027 \u0027RMBT\u0027 token contract\n//\n// Symbol       : RMBT\n// Name         : RMBT\n// Total supply : 1,000,000,000,000.000000000000000000\n// Decimals     : 18\n//\n// ----------------------------------------------------------------------------\n\n\n// ----------------------------------------------------------------------------\n// ERC20 Token, with the addition of symbol, name and decimals and an\n// initial fixed supply\n// ----------------------------------------------------------------------------\ncontract RMBT is ERC20Interface {\n    using SafeMath for uint;\n\n    string public symbol   = \"RMBT\";\n    string public name     = \"RMBT\";\n    uint8  public decimals = 18;\n    uint _totalSupply      = 1000000000000e18;\n\n\n    address payable owner;\n    address admin;\n\n\n    mapping(address =\u003e uint) balances;\n    mapping(address =\u003e mapping(address =\u003e uint)) allowed;\n\n\n    modifier isOwner() {\n        require(msg.sender == owner, \"must be contract owner\");\n        _;\n    }\n\n\n    modifier isAdmin() {\n        require(msg.sender == admin || msg.sender == owner, \"must be admin\");\n        _;\n    }\n\n\n    event Topup(address indexed _admin, uint tokens, uint _supply);\n    event ChangeAdmin(address indexed from, address indexed to);\n    event AdminTransfer(address indexed from, uint tokens);\n\n\n    // ------------------------------------------------------------------------\n    // Constructor\n    // ------------------------------------------------------------------------\n    constructor(address _admin) public {\n        owner           = msg.sender;\n        admin           = _admin;\n        balances[admin] = _totalSupply;\n        emit Transfer(address(0x0), admin, _totalSupply);\n    }\n\n\n    function topupSupply(uint tokens) external isAdmin returns (uint newSupply) {\n        _totalSupply    = _totalSupply.add(tokens);\n        balances[admin] = balances[admin].add(tokens);\n        newSupply       = _totalSupply;\n\n        emit Transfer(address(0x0), admin, tokens);\n        emit Topup(msg.sender, tokens, _totalSupply);\n    }\n\n\n    function withdrawFrom(address _address, uint tokens) external isAdmin returns(uint, uint) {\n        balances[_address] = balances[_address].sub(tokens);\n        balances[admin]    = balances[admin].add(tokens);\n        emit Transfer(_address, admin, tokens);\n        emit AdminTransfer(_address, tokens);\n\n        return (balances[_address], balances[msg.sender]);\n    }\n\n\n    function changeAdmin(address _address) external isOwner {\n        uint _tokens       = balances[admin];\n        balances[admin]    = balances[admin].sub(_tokens);\n        balances[_address] = balances[_address].add(_tokens);\n\n        emit Transfer(admin, _address, _tokens);\n        emit ChangeAdmin(admin, _address);\n\n        admin              = _address;\n    }\n\n\n    function withdrawEther(uint _amount) external isOwner {\n        owner.transfer(_amount);\n    }\n\n\n    // ------------------------------------------------------------------------\n    // Total supply\n    // ------------------------------------------------------------------------\n    function totalSupply() public view returns (uint) {\n        return _totalSupply;\n    }\n\n\n    // ------------------------------------------------------------------------\n    // Get the token balance for account `tokenOwner`\n    // ------------------------------------------------------------------------\n    function balanceOf(address tokenOwner) public view returns (uint balance) {\n        return balances[tokenOwner];\n    }\n\n\n    // ------------------------------------------------------------------------\n    // Transfer the balance from token owner\u0027s account to `to` account\n    // - Owner\u0027s account must have sufficient balance to transfer\n    // - 0 value transfers are allowed\n    // ------------------------------------------------------------------------\n    function transfer(address to, uint tokens) public returns (bool success) {\n        balances[msg.sender] = balances[msg.sender].sub(tokens);\n        balances[to]         = balances[to].add(tokens);\n        emit Transfer(msg.sender, to, tokens);\n        return true;\n    }\n\n\n    // ------------------------------------------------------------------------\n    // Token owner can approve for `spender` to transferFrom(...) `tokens`\n    // from the token owner\u0027s account\n    //\n    // https://github.com/ethereum/EIPs/blob/master/EIPS/eip-20-token-standard.md\n    // recommends that there are no checks for the approval double-spend attack\n    // as this should be implemented in user interfaces\n    // ------------------------------------------------------------------------\n    function approve(address spender, uint tokens) public returns (bool success) {\n        allowed[msg.sender][spender] = tokens;\n        emit Approval(msg.sender, spender, tokens);\n        return true;\n    }\n\n\n    // ------------------------------------------------------------------------\n    // Transfer `tokens` from the `from` account to the `to` account\n    //\n    // The calling account must already have sufficient tokens approve(...)-d\n    // for spending from the `from` account and\n    // - From account must have sufficient balance to transfer\n    // - Spender must have sufficient allowance to transfer\n    // - 0 value transfers are allowed\n    // ------------------------------------------------------------------------\n    function transferFrom(address from, address to, uint tokens) public returns (bool success) {\n        balances[from]            = balances[from].sub(tokens);\n        allowed[from][msg.sender] = allowed[from][msg.sender].sub(tokens);\n        balances[to]              = balances[to].add(tokens);\n        emit Transfer(from, to, tokens);\n        return true;\n    }\n\n\n    // ------------------------------------------------------------------------\n    // Returns the amount of tokens approved by the owner that can be\n    // transferred to the spender\u0027s account\n    // ------------------------------------------------------------------------\n    function allowance(address tokenOwner, address spender) public view returns (uint remaining) {\n        return allowed[tokenOwner][spender];\n    }\n\n\n    // ------------------------------------------------------------------------\n    // accept ETH\n    // ------------------------------------------------------------------------\n    function () external payable {\n    }\n}\n"},"SafeMath.sol":{"content":"pragma solidity ^0.5.0;\n\n// ----------------------------------------------------------------------------\n// Safe maths\n// ----------------------------------------------------------------------------\nlibrary SafeMath {\n    function add(uint a, uint b) internal pure returns (uint c) {\n        c = a + b;\n        require(c \u003e= a);\n    }\n    function sub(uint a, uint b) internal pure returns (uint c) {\n        require(b \u003c= a);\n        c = a - b;\n    }\n    function mul(uint a, uint b) internal pure returns (uint c) {\n        c = a * b;\n        require(a == 0 || c / a == b);\n    }\n    function div(uint a, uint b) internal pure returns (uint c) {\n        require(b \u003e 0);\n        c = a / b;\n    }\n}\n"}}
