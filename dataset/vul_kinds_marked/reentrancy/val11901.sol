pragma solidity 0.4.24;

// File: contracts/generic/Restricted.sol

/*
    Generic contract to authorise calls to certain functions only from a given address.
    The address authorised must be a contract (multisig or not, depending on the permission), except for local test

    deployment works as:
           1. contract deployer account deploys contracts
           2. constructor grants "PermissionGranter" permission to deployer account
           3. deployer account executes initial setup (no multiSig)
           4. deployer account grants PermissionGranter permission for the MultiSig contract
                (e.g. StabilityBoardProxy or PreTokenProxy)
           5. deployer account revokes its own PermissionGranter permission
*/

pragma solidity 0.4.24;


contract Restricted {

    // NB: using bytes32 rather than the string type because it's cheaper gas-wise:
    mapping (address => mapping (bytes32 => bool)) public permissions;

    event PermissionGranted(address indexed agent, bytes32 grantedPermission);
    event PermissionRevoked(address indexed agent, bytes32 revokedPermission);

    modifier restrict(bytes32 requiredPermission) {
        require(permissions[msg.sender][requiredPermission], "msg.sender must have permission");
        _;
    }

    constructor(address permissionGranterContract) public {
        require(permissionGranterContract != address(0), "permissionGranterContract must be set");
        permissions[permissionGranterContract]["PermissionGranter"] = true;
        emit PermissionGranted(permissionGranterContract, "PermissionGranter");
    }

    function grantPermission(address agent, bytes32 requiredPermission) public {
        require(permissions[msg.sender]["PermissionGranter"],
            "msg.sender must have PermissionGranter permission");
        permissions[agent][requiredPermission] = true;
        emit PermissionGranted(agent, requiredPermission);
    }

    function grantMultiplePermissions(address agent, bytes32[] requiredPermissions) public {
        require(permissions[msg.sender]["PermissionGranter"],
            "msg.sender must have PermissionGranter permission");
        uint256 length = requiredPermissions.length;
        for (uint256 i = 0; i < length; i++) {
            grantPermission(agent, requiredPermissions[i]);
        }
    }

    function revokePermission(address agent, bytes32 requiredPermission) public {
        require(permissions[msg.sender]["PermissionGranter"],
            "msg.sender must have PermissionGranter permission");
        permissions[agent][requiredPermission] = false;
        emit PermissionRevoked(agent, requiredPermission);
    }

    function revokeMultiplePermissions(address agent, bytes32[] requiredPermissions) public {
        uint256 length = requiredPermissions.length;
        for (uint256 i = 0; i < length; i++) {
            revokePermission(agent, requiredPermissions[i]);
        }
    }

}

// File: contracts/generic/SafeMath.sol

/**
* @title SafeMath
* @dev Math operations with safety checks that throw on error

    TODO: check against ds-math: https://blog.dapphub.com/ds-math/
    TODO: move roundedDiv to a sep lib? (eg. Math.sol)
    TODO: more unit tests!
*/
pragma solidity 0.4.24;


library SafeMath {
    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a * b;
        require(a == 0 || c / a == b, "mul overflow");
        return c;
    }

    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b > 0, "div by 0"); // Solidity automatically throws for div by 0 but require to emit reason
        uint256 c = a / b;
        // require(a == b * c + a % b); // There is no case in which this doesn't hold
        return c;
    }

    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b <= a, "sub underflow");
        return a - b;
    }

    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        require(c >= a, "add overflow");
        return c;
    }

    // Division, round to nearest integer, round half up
    function roundedDiv(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b > 0, "div by 0"); // Solidity automatically throws for div by 0 but require to emit reason
        uint256 halfB = (b % 2 == 0) ? (b / 2) : (b / 2 + 1);
        return (a % b >= halfB) ? (a / b + 1) : (a / b);
    }

    // Division, always rounds up
    function ceilDiv(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b > 0, "div by 0"); // Solidity automatically throws for div by 0 but require to emit reason
        return (a % b != 0) ? (a / b + 1) : (a / b);
    }

    function min(uint256 a, uint256 b) internal pure returns (uint256) {
        return a < b ? a : b;
    }

    function max(uint256 a, uint256 b) internal pure returns (uint256) {
        return a < b ? b : a;
    }    
}

// File: contracts/interfaces/TransferFeeInterface.sol

/*
 *  transfer fee calculation interface
 *
 */
pragma solidity 0.4.24;


interface TransferFeeInterface {
    function calculateTransferFee(address from, address to, uint amount) external view returns (uint256 fee);
}

// File: contracts/interfaces/ERC20Interface.sol

/*
 * ERC20 interface
 * see https://github.com/ethereum/EIPs/issues/20
 */
pragma solidity 0.4.24;


interface ERC20Interface {
    event Approval(address indexed _owner, address indexed _spender, uint _value);
    event Transfer(address indexed from, address indexed to, uint amount);

    function transfer(address to, uint value) external returns (bool); // solhint-disable-line no-simple-event-func-name
    function transferFrom(address from, address to, uint value) external returns (bool);
    function approve(address spender, uint value) external returns (bool);
    function balanceOf(address who) external view returns (uint);
    function allowance(address _owner, address _spender) external view returns (uint remaining);

}

// File: contracts/interfaces/TokenReceiver.sol

/*
 *  receiver contract interface
 * see https://github.com/ethereum/EIPs/issues/677
 */
pragma solidity 0.4.24;


interface TokenReceiver {
    function transferNotification(address from, uint256 amount, uint data) external;
}

// File: contracts/interfaces/AugmintTokenInterface.sol

/* Augmint Token interface (abstract contract)

TODO: overload transfer() & transferFrom() instead of transferWithNarrative() & transferFromWithNarrative()
      when this fix available in web3& truffle also uses that web3: https://github.com/ethereum/web3.js/pull/1185
TODO: shall we use bytes for narrative?
 */
pragma solidity 0.4.24;







contract AugmintTokenInterface is Restricted, ERC20Interface {
    using SafeMath for uint256;

    string public name;
    string public symbol;
    bytes32 public peggedSymbol;
    uint8 public decimals;

    uint public totalSupply;
    mapping(address => uint256) public balances; // Balances for each account
    mapping(address => mapping (address => uint256)) public allowed; // allowances added with approve()

    TransferFeeInterface public feeAccount;
    mapping(bytes32 => bool) public delegatedTxHashesUsed; // record txHashes used by delegatedTransfer

    event TransferFeesChanged(uint transferFeePt, uint transferFeeMin, uint transferFeeMax);
    event Transfer(address indexed from, address indexed to, uint amount);
    event AugmintTransfer(address indexed from, address indexed to, uint amount, string narrative, uint fee);
    event TokenIssued(uint amount);
    event TokenBurned(uint amount);
    event Approval(address indexed _owner, address indexed _spender, uint256 _value);

    function transfer(address to, uint value) external returns (bool); // solhint-disable-line no-simple-event-func-name
    function transferFrom(address from, address to, uint value) external returns (bool);
    function approve(address spender, uint value) external returns (bool);

    function delegatedTransfer(address from, address to, uint amount, string narrative,
                                    uint maxExecutorFeeInToken, /* client provided max fee for executing the tx */
                                    bytes32 nonce, /* random nonce generated by client */
                                    /* ^^^^ end of signed data ^^^^ */
                                    bytes signature,
                                    uint requestedExecutorFeeInToken /* the executor can decide to request lower fee */
                                ) external;

    function delegatedTransferAndNotify(address from, TokenReceiver target, uint amount, uint data,
                                    uint maxExecutorFeeInToken, /* client provided max fee for executing the tx */
                                    bytes32 nonce, /* random nonce generated by client */
                                    /* ^^^^ end of signed data ^^^^ */
                                    bytes signature,
                                    uint requestedExecutorFeeInToken /* the executor can decide to request lower fee */
                                ) external;

    function increaseApproval(address spender, uint addedValue) external;
    function decreaseApproval(address spender, uint subtractedValue) external;

    function issueTo(address to, uint amount) external; // restrict it to "MonetarySupervisor" in impl.;
    function burn(uint amount) external;

    function transferAndNotify(TokenReceiver target, uint amount, uint data) external;

    function transferWithNarrative(address to, uint256 amount, string narrative) external;
    function transferFromWithNarrative(address from, address to, uint256 amount, string narrative) external;

    function setName(string _name) external;
    function setSymbol(string _symbol) external;

    function allowance(address owner, address spender) external view returns (uint256 remaining);

    function balanceOf(address who) external view returns (uint);


}

// File: contracts/generic/ECRecovery.sol

/**
 * @title Eliptic curve signature operations
 *
 * @dev Based on https://github.com/OpenZeppelin/openzeppelin-solidity/blob/master/contracts/ECRecovery.sol
 *
 * TODO Remove this library once solidity supports passing a signature to ecrecover.
 * See https://github.com/ethereum/solidity/issues/864
 *
 */

library ECRecovery {

  /**
   * @dev Recover signer address from a message by using their signature
   * @param hash bytes32 message, the hash is the signed message. What is recovered is the signer address.
   * @param sig bytes signature, the signature is generated using web3.eth.sign()
   */
  function recover(bytes32 hash, bytes sig)
    internal
    pure
    returns (address)
  {
    bytes32 r;
    bytes32 s;
    uint8 v;

    // Check the signature length
    if (sig.length != 65) {
      return (address(0));
    }

    // Divide the signature in r, s and v variables
    // ecrecover takes the signature parameters, and the only way to get them
    // currently is to use assembly.
    // solium-disable-next-line security/no-inline-assembly
    assembly {
      r := mload(add(sig, 32))
      s := mload(add(sig, 64))
      v := byte(0, mload(add(sig, 96)))
    }

    // Version of signature should be 27 or 28, but 0 and 1 are also possible versions
    if (v < 27) {
      v += 27;
    }

    // If the version is correct return the signer address
    if (v != 27 && v != 28) {
      return (address(0));
    } else {
      // solium-disable-next-line arg-overflow
      return ecrecover(hash, v, r, s);
    }
  }

  /**
   * toEthSignedMessageHash
   * @dev prefix a bytes32 value with "\x19Ethereum Signed Message:"
   * @dev and hash the result
   */
  function toEthSignedMessageHash(bytes32 hash)
    internal
    pure
    returns (bytes32)
  {
    // 32 is the length in bytes of hash,
    // enforced by the type signature above
    return keccak256(abi.encodePacked("\x19Ethereum Signed Message:\n32", hash));
  }
}

// File: contracts/generic/AugmintToken.sol

/* Generic Augmint Token implementation (ERC20 token)
    This contract manages:
        * Balances of Augmint holders and transactions between them
        * Issues/burns tokens

    TODO:
        - reconsider delegatedTransfer and how to structure it
        - shall we allow change of txDelegator?
        - consider generic bytes arg instead of uint for transferAndNotify
        - consider separate transfer fee params and calculation to separate contract (to feeAccount?)
*/
pragma solidity 0.4.24;






contract AugmintToken is AugmintTokenInterface {

    event FeeAccountChanged(TransferFeeInterface newFeeAccount);

    constructor(address permissionGranterContract, string _name, string _symbol, bytes32 _peggedSymbol, uint8 _decimals, TransferFeeInterface _feeAccount)
    public Restricted(permissionGranterContract) {
        require(_feeAccount != address(0), "feeAccount must be set");
        require(bytes(_name).length > 0, "name must be set");
        require(bytes(_symbol).length > 0, "symbol must be set");

        name = _name;
        symbol = _symbol;
        peggedSymbol = _peggedSymbol;
        decimals = _decimals;

        feeAccount = _feeAccount;

    }

    function transfer(address to, uint256 amount) external returns (bool) {
        _transfer(msg.sender, to, amount, "");
        return true;
    }

    /* Transfers based on an offline signed transfer instruction. */
    function delegatedTransfer(address from, address to, uint amount, string narrative,
                                     uint maxExecutorFeeInToken, /* client provided max fee for executing the tx */
                                     bytes32 nonce, /* random nonce generated by client */
                                     /* ^^^^ end of signed data ^^^^ */
                                     bytes signature,
                                     uint requestedExecutorFeeInToken /* the executor can decide to request lower fee */
                                     )
    external {
        bytes32 txHash = keccak256(abi.encodePacked(this, from, to, amount, narrative, maxExecutorFeeInToken, nonce));

        _checkHashAndTransferExecutorFee(txHash, signature, from, maxExecutorFeeInToken, requestedExecutorFeeInToken);

        _transfer(from, to, amount, narrative);
    }

    function approve(address _spender, uint256 amount) external returns (bool) {
        require(_spender != 0x0, "spender must be set");
        allowed[msg.sender][_spender] = amount;
        emit Approval(msg.sender, _spender, amount);
        return true;
    }

    /**
     ERC20 transferFrom attack protection: https://github.com/DecentLabs/dcm-poc/issues/57
     approve should be called when allowed[_spender] == 0. To increment allowed value is better
     to use this function to avoid 2 calls (and wait until the first transaction is mined)
     Based on MonolithDAO Token.sol */
    function increaseApproval(address _spender, uint _addedValue) external {
        require(_spender != 0x0, "spender must be set");
        mapping (address => uint256) allowances = allowed[msg.sender];
        uint newValue = allowances[_spender].add(_addedValue);
        allowances[_spender] = newValue;
        emit Approval(msg.sender, _spender, newValue);
    }

    function decreaseApproval(address _spender, uint _subtractedValue) external {
        require(_spender != 0x0, "spender must be set");
        uint oldValue = allowed[msg.sender][_spender];
        if (_subtractedValue > oldValue) {
            allowed[msg.sender][_spender] = 0;
        } else {
            allowed[msg.sender][_spender] = oldValue.sub(_subtractedValue);
        }
        emit Approval(msg.sender, _spender, allowed[msg.sender][_spender]);
    }

    function transferFrom(address from, address to, uint256 amount) external returns (bool) {
        _transferFrom(from, to, amount, "");
        return true;
    }

    // Issue tokens. See MonetarySupervisor but as a rule of thumb issueTo is only allowed:
    //      - on new loan (by trusted Lender contracts)
    //      - when converting old tokens using MonetarySupervisor
    //      - strictly to reserve by Stability Board (via MonetarySupervisor)
    function issueTo(address to, uint amount) external restrict("MonetarySupervisor") {
        balances[to] = balances[to].add(amount);
        totalSupply = totalSupply.add(amount);
        emit Transfer(0x0, to, amount);
        emit AugmintTransfer(0x0, to, amount, "", 0);
    }

    // Burn tokens. Anyone can burn from its own account. YOLO.
    // Used by to burn from Augmint reserve or by Lender contract after loan repayment
    function burn(uint amount) external {
        require(balances[msg.sender] >= amount, "balance must be >= amount");
        balances[msg.sender] = balances[msg.sender].sub(amount);
        totalSupply = totalSupply.sub(amount);
        emit Transfer(msg.sender, 0x0, amount);
        emit AugmintTransfer(msg.sender, 0x0, amount, "", 0);
    }

    /* to upgrade feeAccount (eg. for fee calculation changes) */
    function setFeeAccount(TransferFeeInterface newFeeAccount) external restrict("StabilityBoard") {
        feeAccount = newFeeAccount;
        emit FeeAccountChanged(newFeeAccount);
    }

    /*  transferAndNotify can be used by contracts which require tokens to have only 1 tx (instead of approve + call)
        Eg. repay loan, lock funds, token sell order on exchange
        Reverts on failue:
            - transfer fails
            - if transferNotification fails (callee must revert on failure)
            - if targetContract is an account or targetContract doesn't have neither transferNotification or fallback fx
        TODO: make data param generic bytes (see receiver code attempt in Locker.transferNotification)
    */
    function transferAndNotify(TokenReceiver target, uint amount, uint data) external {
        _transfer(msg.sender, target, amount, "");

        target.transferNotification(msg.sender, amount, data);
    }

    /* transferAndNotify based on an  instruction signed offline  */
    function delegatedTransferAndNotify(address from, TokenReceiver target, uint amount, uint data,
                                     uint maxExecutorFeeInToken, /* client provided max fee for executing the tx */
                                     bytes32 nonce, /* random nonce generated by client */
                                     /* ^^^^ end of signed data ^^^^ */
                                     bytes signature,
                                     uint requestedExecutorFeeInToken /* the executor can decide to request lower fee */
                                     )
    external {
        bytes32 txHash = keccak256(abi.encodePacked(this, from, target, amount, data, maxExecutorFeeInToken, nonce));

        _checkHashAndTransferExecutorFee(txHash, signature, from, maxExecutorFeeInToken, requestedExecutorFeeInToken);

        _transfer(from, target, amount, "");
        target.transferNotification(from, amount, data);
    }


    function transferWithNarrative(address to, uint256 amount, string narrative) external {
        _transfer(msg.sender, to, amount, narrative);
    }

    function transferFromWithNarrative(address from, address to, uint256 amount, string narrative) external {
        _transferFrom(from, to, amount, narrative);
    }

    /* Allow Stability Board to change the name when a new token contract version
       is deployed and ready for production use. So that older token contracts
       are identifiable in 3rd party apps. */
    function setName(string _name) external restrict("StabilityBoard") {
        name = _name;
    }

    /* Allow Stability Board to change the symbol when a new token contract version
       is deployed and ready for production use. So that older token contracts
       are identifiable in 3rd party apps. */
    function setSymbol(string _symbol) external restrict("StabilityBoard") {
        symbol = _symbol;
    }

    function balanceOf(address _owner) external view returns (uint256 balance) {
        return balances[_owner];
    }

    function allowance(address _owner, address _spender) external view returns (uint256 remaining) {
        return allowed[_owner][_spender];
    }

    function _checkHashAndTransferExecutorFee(bytes32 txHash, bytes signature, address signer,
                                                uint maxExecutorFeeInToken, uint requestedExecutorFeeInToken) private {
        require(requestedExecutorFeeInToken <= maxExecutorFeeInToken, "requestedExecutorFee must be <= maxExecutorFee");
        require(!delegatedTxHashesUsed[txHash], "txHash already used");
        delegatedTxHashesUsed[txHash] = true;

        address recovered = ECRecovery.recover(ECRecovery.toEthSignedMessageHash(txHash), signature);
        require(recovered == signer, "invalid signature");

        _transfer(signer, msg.sender, requestedExecutorFeeInToken, "Delegated transfer fee", 0);
    }

    function _transferFrom(address from, address to, uint256 amount, string narrative) private {
        uint fee = feeAccount.calculateTransferFee(from, to, amount);
        uint amountWithFee = amount.add(fee);

        /* NB: fee is deducted from owner, so transferFrom could fail
            if amount + fee is not available on owner balance, or allowance */
        require(balances[from] >= amountWithFee, "balance must be >= amount + fee");
        require(allowed[from][msg.sender] >= amountWithFee, "allowance must be >= amount + fee");

        _transfer(from, to, amount, narrative, fee);

        allowed[from][msg.sender] = allowed[from][msg.sender].sub(amountWithFee);
    }

    function _transfer(address from, address to, uint transferAmount, string narrative) private {
        uint fee = feeAccount.calculateTransferFee(from, to, transferAmount);
        _transfer(from, to, transferAmount, narrative, fee);
    }

    function _transfer(address from, address to, uint transferAmount, string narrative, uint fee) private {
        require(to != 0x0, "to must be set");
        uint amountWithFee = transferAmount.add(fee);
        // to emit proper reason instead of failing on from.sub()
        require(balances[from] >= amountWithFee, "balance must be >= amount + transfer fee");

        balances[from] = balances[from].sub(amountWithFee);
        balances[to] = balances[to].add(transferAmount);

        emit Transfer(from, to, transferAmount);

        if (fee > 0) {
            balances[feeAccount] = balances[feeAccount].add(fee);
            emit Transfer(from, feeAccount, fee);
        }

        emit AugmintTransfer(from, to, transferAmount, narrative, fee);
    }
}

// File: contracts/generic/SystemAccount.sol

/* Contract to collect fees from system */
pragma solidity 0.4.24;




contract SystemAccount is Restricted {
    event WithdrawFromSystemAccount(address tokenAddress, address to, uint tokenAmount, uint weiAmount,
                                    string narrative);

    constructor(address permissionGranterContract)
    public Restricted(permissionGranterContract) {} // solhint-disable-line no-empty-blocks

    function withdraw(AugmintToken tokenAddress, address to, uint tokenAmount, uint weiAmount, string narrative)
    external restrict("StabilityBoard") {
        tokenAddress.transferWithNarrative(to, tokenAmount, narrative);
        if (weiAmount > 0) {
            to.transfer(weiAmount);
        }
        emit WithdrawFromSystemAccount(tokenAddress, to, tokenAmount, weiAmount, narrative);
    }
}

// File: contracts/AugmintReserves.sol

/* Contract to hold Augmint reserves (ETH & Token)
    - ETH as regular ETH balance of the contract
    - ERC20 token reserve (stored as regular Token balance under the contract address)

NB: reserves are held under the contract address, therefore any transaction on the reserve is limited to the
    tx-s defined here (i.e. transfer is not allowed even by the contract owner or StabilityBoard or MonetarySupervisor)

 */

pragma solidity 0.4.24;




contract AugmintReserves is Restricted {

    event ReserveMigration(address to, uint weiAmount);

    constructor(address permissionGranterContract)
    public Restricted(permissionGranterContract) {} // solhint-disable-line no-empty-blocks

    function () external payable { // solhint-disable-line no-empty-blocks
        // to accept ETH sent into reserve (from defaulted loan's collateral )
    }

    function burn(AugmintTokenInterface augmintToken, uint amount)
    external restrict("MonetarySupervisor") {
        augmintToken.burn(amount);
    }

    function migrate(address to, uint weiAmount)
    external restrict("StabilityBoard") {
        if (weiAmount > 0) {
            to.transfer(weiAmount);
        }
        emit ReserveMigration(to, weiAmount);
    }
}

// File: contracts/Rates.sol

/*
 Generic symbol / WEI rates contract.
 only callable by trusted price oracles.
 Being regularly called by a price oracle
    TODO: trustless/decentrilezed price Oracle
    TODO: shall we use blockNumber instead of now for lastUpdated?
    TODO: consider if we need storing rates with variable decimals instead of fixed 4
    TODO: could we emit 1 RateChanged event from setMultipleRates (symbols and newrates arrays)?
*/
pragma solidity 0.4.24;




contract Rates is Restricted {
    using SafeMath for uint256;

    struct RateInfo {
        uint rate; // how much 1 WEI worth 1 unit , i.e. symbol/ETH rate
                    // 0 rate means no rate info available
        uint lastUpdated;
    }

    // mapping currency symbol => rate. all rates are stored with 2 decimals. i.e. EUR/ETH = 989.12 then rate = 98912
    mapping(bytes32 => RateInfo) public rates;

    event RateChanged(bytes32 symbol, uint newRate);

    constructor(address permissionGranterContract) public Restricted(permissionGranterContract) {} // solhint-disable-line no-empty-blocks

    function setRate(bytes32 symbol, uint newRate) external restrict("RatesFeeder") {
        rates[symbol] = RateInfo(newRate, now);
        emit RateChanged(symbol, newRate);
    }

    function setMultipleRates(bytes32[] symbols, uint[] newRates) external restrict("RatesFeeder") {
        require(symbols.length == newRates.length, "symobls and newRates lengths must be equal");
        for (uint256 i = 0; i < symbols.length; i++) {
            rates[symbols[i]] = RateInfo(newRates[i], now);
            emit RateChanged(symbols[i], newRates[i]);
        }
    }

    function convertFromWei(bytes32 bSymbol, uint weiValue) external view returns(uint value) {
        require(rates[bSymbol].rate > 0, "rates[bSymbol] must be > 0");
        return weiValue.mul(rates[bSymbol].rate).roundedDiv(1000000000000000000);
    }

    function convertToWei(bytes32 bSymbol, uint value) external view returns(uint weiValue) {
        // next line would revert with div by zero but require to emit reason
        require(rates[bSymbol].rate > 0, "rates[bSymbol] must be > 0");
        /* TODO: can we make this not loosing max scale? */
        return value.mul(1000000000000000000).roundedDiv(rates[bSymbol].rate);
    }

}

// File: contracts/Exchange.sol

/* Augmint's Internal Exchange

  For flows see: https://github.com/Augmint/augmint-contracts/blob/master/docs/exchangeFlow.png

    TODO:
        - change to wihtdrawal pattern, see: https://github.com/Augmint/augmint-contracts/issues/17
        - deduct fee
        - consider take funcs (frequent rate changes with takeBuyToken? send more and send back remainder?)
        - use Rates interface?
*/
pragma solidity 0.4.24;






contract Exchange is Restricted {
    using SafeMath for uint256;

    AugmintTokenInterface public augmintToken;
    Rates public rates;

    struct Order {
        uint64 index;
        address maker;

        // % of published current peggedSymbol/ETH rates published by Rates contract. Stored as parts per million
        // I.e. 1,000,000 = 100% (parity), 990,000 = 1% below parity
        uint32 price;

        // buy order: amount in wei
        // sell order: token amount
        uint amount;
    }

    uint64 public orderCount;
    mapping(uint64 => Order) public buyTokenOrders;
    mapping(uint64 => Order) public sellTokenOrders;

    uint64[] private activeBuyOrders;
    uint64[] private activeSellOrders;

    /* used to stop executing matchMultiple when running out of gas.
        actual is much less, just leaving enough matchMultipleOrders() to finish TODO: fine tune & test it*/
    uint32 private constant ORDER_MATCH_WORST_GAS = 100000;

    event NewOrder(uint64 indexed orderId, address indexed maker, uint32 price, uint tokenAmount, uint weiAmount);

    event OrderFill(address indexed tokenBuyer, address indexed tokenSeller, uint64 buyTokenOrderId,
        uint64 sellTokenOrderId, uint publishedRate, uint32 price, uint weiAmount, uint tokenAmount);

    event CancelledOrder(uint64 indexed orderId, address indexed maker, uint tokenAmount, uint weiAmount);

    event RatesContractChanged(Rates newRatesContract);

    constructor(address permissionGranterContract, AugmintTokenInterface _augmintToken, Rates _rates)
    public Restricted(permissionGranterContract) {
        augmintToken = _augmintToken;
        rates = _rates;
    }

    /* to allow upgrade of Rates  contract */
    function setRatesContract(Rates newRatesContract)
    external restrict("StabilityBoard") {
        rates = newRatesContract;
        emit RatesContractChanged(newRatesContract);
    }

    function placeBuyTokenOrder(uint32 price) external payable returns (uint64 orderId) {
        require(price > 0, "price must be > 0");
        require(msg.value > 0, "msg.value must be > 0");

        orderId = ++orderCount;
        buyTokenOrders[orderId] = Order(uint64(activeBuyOrders.length), msg.sender, price, msg.value);
        activeBuyOrders.push(orderId);

        emit NewOrder(orderId, msg.sender, price, 0, msg.value);
    }

    /* this function requires previous approval to transfer tokens */
    function placeSellTokenOrder(uint32 price, uint tokenAmount) external returns (uint orderId) {
        augmintToken.transferFrom(msg.sender, this, tokenAmount);
        return _placeSellTokenOrder(msg.sender, price, tokenAmount);
    }

    /* place sell token order called from AugmintToken's transferAndNotify
     Flow:
        1) user calls token contract's transferAndNotify price passed in data arg
        2) transferAndNotify transfers tokens to the Exchange contract
        3) transferAndNotify calls Exchange.transferNotification with lockProductId
    */
    function transferNotification(address maker, uint tokenAmount, uint price) external {
        require(msg.sender == address(augmintToken), "msg.sender must be augmintToken");
        _placeSellTokenOrder(maker, uint32(price), tokenAmount);
    }

    function cancelBuyTokenOrder(uint64 buyTokenId) external {
        Order storage order = buyTokenOrders[buyTokenId];
        require(order.maker == msg.sender, "msg.sender must be order.maker");
        require(order.amount > 0, "buy order already removed");

        uint amount = order.amount;
        order.amount = 0;
        _removeBuyOrder(order);

        msg.sender.transfer(amount);

        emit CancelledOrder(buyTokenId, msg.sender, 0, amount);
    }

    function cancelSellTokenOrder(uint64 sellTokenId) external {
        Order storage order = sellTokenOrders[sellTokenId];
        require(order.maker == msg.sender, "msg.sender must be order.maker");
        require(order.amount > 0, "sell order already removed");

        uint amount = order.amount;
        order.amount = 0;
        _removeSellOrder(order);

        augmintToken.transferWithNarrative(msg.sender, amount, "Sell token order cancelled");

        emit CancelledOrder(sellTokenId, msg.sender, amount, 0);
    }

    /* matches any two orders if the sell price >= buy price
        trade price is the price of the maker (the order placed earlier)
        reverts if any of the orders have been removed
    */
    function matchOrders(uint64 buyTokenId, uint64 sellTokenId) external {
        require(_fillOrder(buyTokenId, sellTokenId), "fill order failed");
    }

    /*  matches as many orders as possible from the passed orders
        Runs as long as gas is available for the call.
        Reverts if any match is invalid (e.g sell price > buy price)
        Skips match if any of the matched orders is removed / already filled (i.e. amount = 0)
    */
    function matchMultipleOrders(uint64[] buyTokenIds, uint64[] sellTokenIds) external returns(uint matchCount) {
        uint len = buyTokenIds.length;
        require(len == sellTokenIds.length, "buyTokenIds and sellTokenIds lengths must be equal");

        for (uint i = 0; i < len && gasleft() > ORDER_MATCH_WORST_GAS; i++) {
            if(_fillOrder(buyTokenIds[i], sellTokenIds[i])) {
                matchCount++;
            }
        }
    }

    function getActiveOrderCounts() external view returns(uint buyTokenOrderCount, uint sellTokenOrderCount) {
        return(activeBuyOrders.length, activeSellOrders.length);
    }

    // returns <chunkSize> active buy orders starting from <offset>
    // orders are encoded as [id, maker, price, amount]
    function getActiveBuyOrders(uint offset, uint16 chunkSize)
    external view returns (uint[4][]) {
        uint limit = SafeMath.min(offset.add(chunkSize), activeBuyOrders.length);
        uint[4][] memory response = new uint[4][](limit.sub(offset));
        for (uint i = offset; i < limit; i++) {
            uint64 orderId = activeBuyOrders[i];
            Order storage order = buyTokenOrders[orderId];
            response[i - offset] = [orderId, uint(order.maker), order.price, order.amount];
        }
        return response;
    }

    // returns <chunkSize> active sell orders starting from <offset>
    // orders are encoded as [id, maker, price, amount]
    function getActiveSellOrders(uint offset, uint16 chunkSize)
    external view returns (uint[4][]) {
        uint limit = SafeMath.min(offset.add(chunkSize), activeSellOrders.length);
        uint[4][] memory response = new uint[4][](limit.sub(offset));
        for (uint i = offset; i < limit; i++) {
            uint64 orderId = activeSellOrders[i];
            Order storage order = sellTokenOrders[orderId];
            response[i - offset] = [orderId, uint(order.maker), order.price, order.amount];
        }
        return response;
    }

    uint private constant E12 = 1000000000000;

    function _fillOrder(uint64 buyTokenId, uint64 sellTokenId) private returns(bool success) {
        Order storage buy = buyTokenOrders[buyTokenId];
        Order storage sell = sellTokenOrders[sellTokenId];
        if( buy.amount == 0 || sell.amount == 0 ) {
            return false; // one order is already filled and removed.
                          // we let matchMultiple continue, indivudal match will revert
        }

        require(buy.price >= sell.price, "buy price must be >= sell price");

        // pick maker's price (whoever placed order sooner considered as maker)
        uint32 price = buyTokenId > sellTokenId ? sell.price : buy.price;

        uint publishedRate;
        (publishedRate, ) = rates.rates(augmintToken.peggedSymbol());
        // fillRate = publishedRate * 1000000 / price

        uint sellWei = sell.amount.mul(uint(price)).mul(E12).roundedDiv(publishedRate);

        uint tradedWei;
        uint tradedTokens;
        if (sellWei <= buy.amount) {
            tradedWei = sellWei;
            tradedTokens = sell.amount;
        } else {
            tradedWei = buy.amount;
            tradedTokens = buy.amount.mul(publishedRate).roundedDiv(uint(price).mul(E12));
        }

        buy.amount = buy.amount.sub(tradedWei);
        if (buy.amount == 0) {
            _removeBuyOrder(buy);
        }

        sell.amount = sell.amount.sub(tradedTokens);
        if (sell.amount == 0) {
            _removeSellOrder(sell);
        }

        augmintToken.transferWithNarrative(buy.maker, tradedTokens, "Buy token order fill");
        sell.maker.transfer(tradedWei);

        emit OrderFill(buy.maker, sell.maker, buyTokenId,
            sellTokenId, publishedRate, price, tradedWei, tradedTokens);

        return true;
    }

    function _placeSellTokenOrder(address maker, uint32 price, uint tokenAmount)
    private returns (uint64 orderId) {
        require(price > 0, "price must be > 0");
        require(tokenAmount > 0, "tokenAmount must be > 0");

        orderId = ++orderCount;
        sellTokenOrders[orderId] = Order(uint64(activeSellOrders.length), maker, price, tokenAmount);
        activeSellOrders.push(orderId);

        emit NewOrder(orderId, maker, price, tokenAmount, 0);
    }

    function _removeBuyOrder(Order storage order) private {
        uint lastIndex = activeBuyOrders.length - 1;
        if (order.index < lastIndex) {
            uint64 movedOrderId = activeBuyOrders[lastIndex];
            activeBuyOrders[order.index] = movedOrderId;
            buyTokenOrders[movedOrderId].index = order.index;
        }
        activeBuyOrders.length--;
    }

    function _removeSellOrder(Order storage order) private {
        uint lastIndex = activeSellOrders.length - 1;
        if (order.index < lastIndex) {
            uint64 movedOrderId = activeSellOrders[lastIndex];
            activeSellOrders[order.index] = movedOrderId;
            sellTokenOrders[movedOrderId].index = order.index;
        }
        activeSellOrders.length--;
    }
}

// File: contracts/FeeAccount.sol

/* Contract to collect fees from system
    TODO: calculateExchangeFee + Exchange params and setters
*/

pragma solidity 0.4.24;





contract FeeAccount is SystemAccount, TransferFeeInterface {

    using SafeMath for uint256;

    struct TransferFee {
        uint pt;  // in parts per million (ppm) , ie. 2,000 = 0.2%
        uint min; // with base unit of augmint token, eg. 2 decimals for token, eg. 310 = 3.1 ACE
        uint max; // with base unit of augmint token, eg. 2 decimals for token, eg. 310 = 3.1 ACE
    }

    TransferFee public transferFee;

    event TransferFeesChanged(uint transferFeePt, uint transferFeeMin, uint transferFeeMax);

    constructor(address permissionGranterContract, uint transferFeePt, uint transferFeeMin, uint transferFeeMax)
    public SystemAccount(permissionGranterContract) {
        transferFee = TransferFee(transferFeePt, transferFeeMin, transferFeeMax);
    }

    function () external payable { // solhint-disable-line no-empty-blocks
        // to accept ETH sent into feeAccount (defaulting fee in ETH )
    }

    function setTransferFees(uint transferFeePt, uint transferFeeMin, uint transferFeeMax)
    external restrict("StabilityBoard") {
        transferFee = TransferFee(transferFeePt, transferFeeMin, transferFeeMax);
        emit TransferFeesChanged(transferFeePt, transferFeeMin, transferFeeMax);
    }

    function calculateTransferFee(address from, address to, uint amount) external view returns (uint256 fee) {
        if (!permissions[from]["NoTransferFee"] && !permissions[to]["NoTransferFee"]) {
            fee = amount.mul(transferFee.pt).div(1000000);
            if (fee > transferFee.max) {
                fee = transferFee.max;
            } else if (fee < transferFee.min) {
                fee = transferFee.min;
            }
        }
        return fee;
    }

    function calculateExchangeFee(uint weiAmount) external view returns (uint256 weiFee) {
        /* TODO: to be implemented and use in Exchange.sol. always revert for now */
        require(weiAmount != weiAmount, "not yet implemented");
        weiFee = transferFee.max; // to silence compiler warnings until it's implemented
    }

}

// File: contracts/InterestEarnedAccount.sol

/* Contract to hold earned interest from loans repaid
   premiums for locks are being accrued (i.e. transferred) to Locker */

pragma solidity 0.4.24;




contract InterestEarnedAccount is SystemAccount {

    constructor(address permissionGranterContract) public SystemAccount(permissionGranterContract) {} // solhint-disable-line no-empty-blocks

    function transferInterest(AugmintTokenInterface augmintToken, address locker, uint interestAmount)
    external restrict("MonetarySupervisor") {
        augmintToken.transfer(locker, interestAmount);
    }

}

// File: contracts/MonetarySupervisor.sol

/* MonetarySupervisor
    - maintains system wide KPIs (eg totalLockAmount, totalLoanAmount)
    - holds system wide parameters/limits
    - enforces system wide limits
    - burns and issues to AugmintReserves
    - Send funds from reserve to exchange when intervening (not implemented yet)
    - Converts older versions of AugmintTokens in 1:1 to new
*/

pragma solidity 0.4.24;








contract MonetarySupervisor is Restricted, TokenReceiver { // solhint-disable-line no-empty-blocks
    using SafeMath for uint256;

    uint public constant PERCENT_100 = 1000000;

    AugmintTokenInterface public augmintToken;
    InterestEarnedAccount public interestEarnedAccount;
    AugmintReserves public augmintReserves;

    uint public issuedByStabilityBoard; // token issued by Stability Board
    uint public burnedByStabilityBoard; // token burned by Stability Board

    uint public totalLoanAmount; // total amount of all loans without interest, in token
    uint public totalLockedAmount; // total amount of all locks without premium, in token

    /**********
        Parameters to ensure totalLoanAmount or totalLockedAmount difference is within limits and system also works
        when total loan or lock amounts are low.
            for test calculations: https://docs.google.com/spreadsheets/d/1MeWYPYZRIm1n9lzpvbq8kLfQg1hhvk5oJY6NrR401S0
    **********/
    struct LtdParams {
        uint  lockDifferenceLimit; /* only allow a new lock if Loan To Deposit ratio would stay above
                                            (1 - lockDifferenceLimit) with new lock. Stored as parts per million */
        uint  loanDifferenceLimit; /* only allow a new loan if Loan To Deposit ratio would stay above
                                            (1 + loanDifferenceLimit) with new loan. Stored as parts per million */
        /* allowedDifferenceAmount param is to ensure the system is not "freezing" when totalLoanAmount or
            totalLockAmount is low.
        It allows a new loan or lock (up to an amount to reach this difference) even if LTD will go below / above
            lockDifferenceLimit / loanDifferenceLimit with the new lock/loan */
        uint  allowedDifferenceAmount;
    }

    LtdParams public ltdParams;

    /* Previously deployed AugmintTokens which are accepted for conversion (see transferNotification() )
        NB: it's not iterable so old version addresses needs to be added for UI manually after each deploy */
    mapping(address => bool) public acceptedLegacyAugmintTokens;

    event LtdParamsChanged(uint lockDifferenceLimit, uint loanDifferenceLimit, uint allowedDifferenceAmount);

    event AcceptedLegacyAugmintTokenChanged(address augmintTokenAddress, bool newAcceptedState);

    event LegacyTokenConverted(address oldTokenAddress, address account, uint amount);

    event KPIsAdjusted(uint totalLoanAmountAdjustment, uint totalLockedAmountAdjustment);

    event SystemContractsChanged(InterestEarnedAccount newInterestEarnedAccount, AugmintReserves newAugmintReserves);

    constructor(address permissionGranterContract, AugmintTokenInterface _augmintToken,
        AugmintReserves _augmintReserves, InterestEarnedAccount _interestEarnedAccount,
        uint lockDifferenceLimit, uint loanDifferenceLimit, uint allowedDifferenceAmount)
    public Restricted(permissionGranterContract) {
        augmintToken = _augmintToken;
        augmintReserves = _augmintReserves;
        interestEarnedAccount = _interestEarnedAccount;

        ltdParams = LtdParams(lockDifferenceLimit, loanDifferenceLimit, allowedDifferenceAmount);
    }

    function issueToReserve(uint amount) external restrict("StabilityBoard") {
        issuedByStabilityBoard = issuedByStabilityBoard.add(amount);
        augmintToken.issueTo(augmintReserves, amount);
    }

    function burnFromReserve(uint amount) external restrict("StabilityBoard") {
        burnedByStabilityBoard = burnedByStabilityBoard.add(amount);
        augmintReserves.burn(augmintToken, amount);
    }

    /* Locker requesting interest when locking funds. Enforcing LTD to stay within range allowed by LTD params
        NB: it does not know about min loan amount, it's the loan contract's responsibility to enforce it  */
    function requestInterest(uint amountToLock, uint interestAmount) external {
        // only whitelisted Locker
        require(permissions[msg.sender]["Locker"], "msg.sender must have Locker permission");
        require(amountToLock <= getMaxLockAmountAllowedByLtd(), "amountToLock must be <= maxLockAmountAllowedByLtd");

        totalLockedAmount = totalLockedAmount.add(amountToLock);
        // next line would revert but require to emit reason:
        require(augmintToken.balanceOf(address(interestEarnedAccount)) >= interestAmount,
            "interestEarnedAccount balance must be >= interestAmount");
        interestEarnedAccount.transferInterest(augmintToken, msg.sender, interestAmount); // transfer interest to Locker
    }

    // Locker notifying when releasing funds to update KPIs
    function releaseFundsNotification(uint lockedAmount) external {
        // only whitelisted Locker
        require(permissions[msg.sender]["Locker"], "msg.sender must have Locker permission");
        totalLockedAmount = totalLockedAmount.sub(lockedAmount);
    }

    /* Issue loan if LTD stays within range allowed by LTD params
        NB: it does not know about min loan amount, it's the loan contract's responsibility to enforce it */
    function issueLoan(address borrower, uint loanAmount) external {
         // only whitelisted LoanManager contracts
        require(permissions[msg.sender]["LoanManager"],
            "msg.sender must have LoanManager permission");
        require(loanAmount <= getMaxLoanAmountAllowedByLtd(), "loanAmount must be <= maxLoanAmountAllowedByLtd");
        totalLoanAmount = totalLoanAmount.add(loanAmount);
        augmintToken.issueTo(borrower, loanAmount);
    }

    function loanRepaymentNotification(uint loanAmount) external {
        // only whitelisted LoanManager contracts
        require(permissions[msg.sender]["LoanManager"],
            "msg.sender must have LoanManager permission");
        totalLoanAmount = totalLoanAmount.sub(loanAmount);
    }

    // NB: this is called by Lender contract with the sum of all loans collected in batch
    function loanCollectionNotification(uint totalLoanAmountCollected) external {
        // only whitelisted LoanManager contracts
        require(permissions[msg.sender]["LoanManager"],
            "msg.sender must have LoanManager permission");
        totalLoanAmount = totalLoanAmount.sub(totalLoanAmountCollected);
    }

    function setAcceptedLegacyAugmintToken(address legacyAugmintTokenAddress, bool newAcceptedState)
    external restrict("StabilityBoard") {
        acceptedLegacyAugmintTokens[legacyAugmintTokenAddress] = newAcceptedState;
        emit AcceptedLegacyAugmintTokenChanged(legacyAugmintTokenAddress, newAcceptedState);
    }

    function setLtdParams(uint lockDifferenceLimit, uint loanDifferenceLimit, uint allowedDifferenceAmount)
    external restrict("StabilityBoard") {
        ltdParams = LtdParams(lockDifferenceLimit, loanDifferenceLimit, allowedDifferenceAmount);
        emit LtdParamsChanged(lockDifferenceLimit, loanDifferenceLimit, allowedDifferenceAmount);
    }

    /* function to migrate old totalLoanAmount and totalLockedAmount from old monetarySupervisor contract
        when it's upgraded.
        Set new monetarySupervisor contract in all locker and loanManager contracts before executing this */
    function adjustKPIs(uint totalLoanAmountAdjustment, uint totalLockedAmountAdjustment)
    external restrict("StabilityBoard") {
        totalLoanAmount = totalLoanAmount.add(totalLoanAmountAdjustment);
        totalLockedAmount = totalLockedAmount.add(totalLockedAmountAdjustment);
        emit KPIsAdjusted(totalLoanAmountAdjustment, totalLockedAmountAdjustment);
    }

    /* to allow upgrades of InterestEarnedAccount and AugmintReserves contracts. */
    function setSystemContracts(InterestEarnedAccount newInterestEarnedAccount, AugmintReserves newAugmintReserves)
    external restrict("StabilityBoard") {
        interestEarnedAccount = newInterestEarnedAccount;
        augmintReserves = newAugmintReserves;
        emit SystemContractsChanged(newInterestEarnedAccount, newAugmintReserves);
    }

    /* User can request to convert their tokens from older AugmintToken versions in 1:1
      transferNotification is called from AugmintToken's transferAndNotify
     Flow for converting old tokens:
        1) user calls old token contract's transferAndNotify with the amount to convert,
                addressing the new MonetarySupervisor Contract
        2) transferAndNotify transfers user's old tokens to the current MonetarySupervisor contract's address
        3) transferAndNotify calls MonetarySupervisor.transferNotification
        4) MonetarySupervisor checks if old AugmintToken is permitted
        5) MonetarySupervisor issues new tokens to user's account in current AugmintToken
        6) MonetarySupervisor burns old tokens from own balance
    */
    function transferNotification(address from, uint amount, uint /* data, not used */ ) external {
        AugmintTokenInterface legacyToken = AugmintTokenInterface(msg.sender);
        require(acceptedLegacyAugmintTokens[legacyToken], "msg.sender must be allowed in acceptedLegacyAugmintTokens");

        legacyToken.burn(amount);
        augmintToken.issueTo(from, amount);
        emit LegacyTokenConverted(msg.sender, from, amount);
    }

    /* Helper function for UI.
        Returns max lock amount based on minLockAmount, interestPt, using LTD params & interestEarnedAccount balance */
    function getMaxLockAmount(uint minLockAmount, uint interestPt) external view returns (uint maxLock) {
        uint allowedByEarning = augmintToken.balanceOf(address(interestEarnedAccount)).mul(PERCENT_100).div(interestPt);
        uint allowedByLtd = getMaxLockAmountAllowedByLtd();
        maxLock = allowedByEarning < allowedByLtd ? allowedByEarning : allowedByLtd;
        maxLock = maxLock < minLockAmount ? 0 : maxLock;
    }

    /* Helper function for UI.
        Returns max loan amount based on minLoanAmont using LTD params */
    function getMaxLoanAmount(uint minLoanAmount) external view returns (uint maxLoan) {
        uint allowedByLtd = getMaxLoanAmountAllowedByLtd();
        maxLoan = allowedByLtd < minLoanAmount ? 0 : allowedByLtd;
    }

    /* returns maximum lockable token amount allowed by LTD params. */
    function getMaxLockAmountAllowedByLtd() public view returns(uint maxLockByLtd) {
        uint allowedByLtdDifferencePt = totalLoanAmount.mul(PERCENT_100).div(PERCENT_100
                                            .sub(ltdParams.lockDifferenceLimit));
        allowedByLtdDifferencePt = totalLockedAmount >= allowedByLtdDifferencePt ?
                                        0 : allowedByLtdDifferencePt.sub(totalLockedAmount);

        uint allowedByLtdDifferenceAmount =
            totalLockedAmount >= totalLoanAmount.add(ltdParams.allowedDifferenceAmount) ?
                0 : totalLoanAmount.add(ltdParams.allowedDifferenceAmount).sub(totalLockedAmount);

        maxLockByLtd = allowedByLtdDifferencePt > allowedByLtdDifferenceAmount ?
                                        allowedByLtdDifferencePt : allowedByLtdDifferenceAmount;
    }

    /* returns maximum borrowable token amount allowed by LTD params */
    function getMaxLoanAmountAllowedByLtd() public view returns(uint maxLoanByLtd) {
        uint allowedByLtdDifferencePt = totalLockedAmount.mul(ltdParams.loanDifferenceLimit.add(PERCENT_100))
                                            .div(PERCENT_100);
        allowedByLtdDifferencePt = totalLoanAmount >= allowedByLtdDifferencePt ?
                                        0 : allowedByLtdDifferencePt.sub(totalLoanAmount);

        uint allowedByLtdDifferenceAmount =
            totalLoanAmount >= totalLockedAmount.add(ltdParams.allowedDifferenceAmount) ?
                0 : totalLockedAmount.add(ltdParams.allowedDifferenceAmount).sub(totalLoanAmount);

        maxLoanByLtd = allowedByLtdDifferencePt > allowedByLtdDifferenceAmount ?
                                        allowedByLtdDifferencePt : allowedByLtdDifferenceAmount;
    }
}

// File: contracts/LoanManager.sol

/*
    Contract to manage Augmint token loan contracts backed by ETH
    For flows see: https://github.com/Augmint/augmint-contracts/blob/master/docs/loanFlow.png

    TODO:
        - create MonetarySupervisor interface and use it instead?
        - make data arg generic bytes?
        - make collect() run as long as gas provided allows
*/
pragma solidity 0.4.24;







contract LoanManager is Restricted, TokenReceiver {
    using SafeMath for uint256;

    enum LoanState { Open, Repaid, Defaulted, Collected } // NB: Defaulted state is not stored, only getters calculate

    struct LoanProduct {
        uint minDisbursedAmount; // 0: with decimals set in AugmintToken.decimals
        uint32 term;            // 1
        uint32 discountRate;    // 2: discountRate in parts per million , ie. 10,000 = 1%
        uint32 collateralRatio; // 3: loan token amount / colleteral pegged ccy value
                                //      in parts per million , ie. 10,000 = 1%
        uint32 defaultingFeePt; // 4: % of collateral in parts per million , ie. 50,000 = 5%
        bool isActive;          // 5
    }

    /* NB: we don't need to store loan parameters because loan products can't be altered (only disabled/enabled) */
    struct LoanData {
        uint collateralAmount; // 0
        uint repaymentAmount; // 1
        address borrower; // 2
        uint32 productId; // 3
        LoanState state; // 4
        uint40 maturity; // 5
    }

    LoanProduct[] public products;

    LoanData[] public loans;
    mapping(address => uint[]) public accountLoans;  // owner account address =>  array of loan Ids

    Rates public rates; // instance of ETH/pegged currency rate provider contract
    AugmintTokenInterface public augmintToken; // instance of token contract
    MonetarySupervisor public monetarySupervisor;

    event NewLoan(uint32 productId, uint loanId, address indexed borrower, uint collateralAmount, uint loanAmount,
        uint repaymentAmount, uint40 maturity);

    event LoanProductActiveStateChanged(uint32 productId, bool newState);

    event LoanProductAdded(uint32 productId);

    event LoanRepayed(uint loanId, address borrower);

    event LoanCollected(uint loanId, address indexed borrower, uint collectedCollateral,
        uint releasedCollateral, uint defaultingFee);

    event SystemContractsChanged(Rates newRatesContract, MonetarySupervisor newMonetarySupervisor);

    constructor(address permissionGranterContract, AugmintTokenInterface _augmintToken,
                    MonetarySupervisor _monetarySupervisor, Rates _rates)
    public Restricted(permissionGranterContract) {
        augmintToken = _augmintToken;
        monetarySupervisor = _monetarySupervisor;
        rates = _rates;
    }

    function addLoanProduct(uint32 term, uint32 discountRate, uint32 collateralRatio, uint minDisbursedAmount,
                                uint32 defaultingFeePt, bool isActive)
    external restrict("StabilityBoard") {

        uint _newProductId = products.push(
            LoanProduct(minDisbursedAmount, term, discountRate, collateralRatio, defaultingFeePt, isActive)
        ) - 1;

        uint32 newProductId = uint32(_newProductId);
        require(newProductId == _newProductId, "productId overflow");

        emit LoanProductAdded(newProductId);
    }

    function setLoanProductActiveState(uint32 productId, bool newState)
    external restrict ("StabilityBoard") {
        require(productId < products.length, "invalid productId"); // next line would revert but require to emit reason
        products[productId].isActive = newState;
        emit LoanProductActiveStateChanged(productId, newState);
    }

    function newEthBackedLoan(uint32 productId) external payable {
        require(productId < products.length, "invalid productId"); // next line would revert but require to emit reason
        LoanProduct storage product = products[productId];
        require(product.isActive, "product must be in active state"); // valid product


        // calculate loan values based on ETH sent in with Tx
        uint tokenValue = rates.convertFromWei(augmintToken.peggedSymbol(), msg.value);
        uint repaymentAmount = tokenValue.mul(product.collateralRatio).div(1000000);

        uint loanAmount;
        (loanAmount, ) = calculateLoanValues(product, repaymentAmount);

        require(loanAmount >= product.minDisbursedAmount, "loanAmount must be >= minDisbursedAmount");

        uint expiration = now.add(product.term);
        uint40 maturity = uint40(expiration);
        require(maturity == expiration, "maturity overflow");

        // Create new loan
        uint loanId = loans.push(LoanData(msg.value, repaymentAmount, msg.sender,
                                            productId, LoanState.Open, maturity)) - 1;

        // Store ref to new loan
        accountLoans[msg.sender].push(loanId);

        // Issue tokens and send to borrower
        monetarySupervisor.issueLoan(msg.sender, loanAmount);

        emit NewLoan(productId, loanId, msg.sender, msg.value, loanAmount, repaymentAmount, maturity);
    }

    /* repay loan, called from AugmintToken's transferAndNotify
     Flow for repaying loan:
        1) user calls token contract's transferAndNotify loanId passed in data arg
        2) transferAndNotify transfers tokens to the Lender contract
        3) transferAndNotify calls Lender.transferNotification with lockProductId
    */
    // from arg is not used as we allow anyone to repay a loan:
    function transferNotification(address, uint repaymentAmount, uint loanId) external {
        require(msg.sender == address(augmintToken), "msg.sender must be augmintToken");

        _repayLoan(loanId, repaymentAmount);
    }

    function collect(uint[] loanIds) external {
        /* when there are a lots of loans to be collected then
             the client need to call it in batches to make sure tx won't exceed block gas limit.
         Anyone can call it - can't cause harm as it only allows to collect loans which they are defaulted
         TODO: optimise defaulting fee calculations
        */
        uint totalLoanAmountCollected;
        uint totalCollateralToCollect;
        uint totalDefaultingFee;
        for (uint i = 0; i < loanIds.length; i++) {
            require(loanIds[i] < loans.length, "invalid loanId"); // next line would revert but require to emit reason
            LoanData storage loan = loans[loanIds[i]];
            require(loan.state == LoanState.Open, "loan state must be Open");
            require(now >= loan.maturity, "current time must be later than maturity");
            LoanProduct storage product = products[loan.productId];

            uint loanAmount;
            (loanAmount, ) = calculateLoanValues(product, loan.repaymentAmount);

            totalLoanAmountCollected = totalLoanAmountCollected.add(loanAmount);

            loan.state = LoanState.Collected;

            // send ETH collateral to augmintToken reserve
            uint defaultingFeeInToken = loan.repaymentAmount.mul(product.defaultingFeePt).div(1000000);
            uint defaultingFee = rates.convertToWei(augmintToken.peggedSymbol(), defaultingFeeInToken);
            uint targetCollection = rates.convertToWei(augmintToken.peggedSymbol(),
                    loan.repaymentAmount).add(defaultingFee);

            uint releasedCollateral;
            if (targetCollection < loan.collateralAmount) {
                releasedCollateral = loan.collateralAmount.sub(targetCollection);
                loan.borrower.transfer(releasedCollateral);
            }
            uint collateralToCollect = loan.collateralAmount.sub(releasedCollateral);
            if (defaultingFee >= collateralToCollect) {
                defaultingFee = collateralToCollect;
                collateralToCollect = 0;
            } else {
                collateralToCollect = collateralToCollect.sub(defaultingFee);
            }
            totalDefaultingFee = totalDefaultingFee.add(defaultingFee);

            totalCollateralToCollect = totalCollateralToCollect.add(collateralToCollect);

            emit LoanCollected(loanIds[i], loan.borrower, collateralToCollect.add(defaultingFee),
                    releasedCollateral, defaultingFee);
        }

        if (totalCollateralToCollect > 0) {
            address(monetarySupervisor.augmintReserves()).transfer(totalCollateralToCollect);
        }

        if (totalDefaultingFee > 0) {
            address(augmintToken.feeAccount()).transfer(totalDefaultingFee);
        }

        monetarySupervisor.loanCollectionNotification(totalLoanAmountCollected);// update KPIs

    }

    /* to allow upgrade of Rates and MonetarySupervisor contracts */
    function setSystemContracts(Rates newRatesContract, MonetarySupervisor newMonetarySupervisor)
    external restrict("StabilityBoard") {
        rates = newRatesContract;
        monetarySupervisor = newMonetarySupervisor;
        emit SystemContractsChanged(newRatesContract, newMonetarySupervisor);
    }

    function getProductCount() external view returns (uint) {
        return products.length;
    }

    // returns <chunkSize> loan products starting from some <offset>:
    // [ productId, minDisbursedAmount, term, discountRate, collateralRatio, defaultingFeePt, maxLoanAmount, isActive ]
    function getProducts(uint offset, uint16 chunkSize)
    external view returns (uint[8][]) {
        uint limit = SafeMath.min(offset.add(chunkSize), products.length);
        uint[8][] memory response = new uint[8][](limit.sub(offset));

        for (uint i = offset; i < limit; i++) {
            LoanProduct storage product = products[i];
            response[i - offset] = [i, product.minDisbursedAmount, product.term, product.discountRate,
                    product.collateralRatio, product.defaultingFeePt,
                    monetarySupervisor.getMaxLoanAmount(product.minDisbursedAmount), product.isActive ? 1 : 0 ];
        }
        return response;
    }

    function getLoanCount() external view returns (uint) {
        return loans.length;
    }

    /* returns <chunkSize> loans starting from some <offset>. Loans data encoded as:
        [loanId, collateralAmount, repaymentAmount, borrower, productId,
              state, maturity, disbursementTime, loanAmount, interestAmount] */
    function getLoans(uint offset, uint16 chunkSize)
    external view returns (uint[10][]) {
        uint limit = SafeMath.min(offset.add(chunkSize), loans.length);
        uint[10][] memory response = new uint[10][](limit.sub(offset));

        for (uint i = offset; i < limit; i++) {
            response[i - offset] = getLoanTuple(i);
        }
        return response;
    }

    function getLoanCountForAddress(address borrower) external view returns (uint) {
        return accountLoans[borrower].length;
    }

    /* returns <chunkSize> loans of a given account, starting from some <offset>. Loans data encoded as:
        [loanId, collateralAmount, repaymentAmount, borrower, productId, state, maturity, disbursementTime,
                                                                                    loanAmount, interestAmount ] */
    function getLoansForAddress(address borrower, uint offset, uint16 chunkSize)
    external view returns (uint[10][]) {
        uint[] storage loansForAddress = accountLoans[borrower];
        uint limit = SafeMath.min(offset.add(chunkSize), loansForAddress.length);
        uint[10][] memory response = new uint[10][](limit.sub(offset));

        for (uint i = offset; i < limit; i++) {
            response[i - offset] = getLoanTuple(loansForAddress[i]);
        }
        return response;
    }

    function getLoanTuple(uint loanId) public view returns (uint[10] result) {
        require(loanId < loans.length, "invalid loanId"); // next line would revert but require to emit reason
        LoanData storage loan = loans[loanId];
        LoanProduct storage product = products[loan.productId];

        uint loanAmount;
        uint interestAmount;
        (loanAmount, interestAmount) = calculateLoanValues(product, loan.repaymentAmount);
        uint disbursementTime = loan.maturity - product.term;

        LoanState loanState =
                loan.state == LoanState.Open && now >= loan.maturity ? LoanState.Defaulted : loan.state;

        result = [loanId, loan.collateralAmount, loan.repaymentAmount, uint(loan.borrower),
                loan.productId, uint(loanState), loan.maturity, disbursementTime, loanAmount, interestAmount];
    }

    function calculateLoanValues(LoanProduct storage product, uint repaymentAmount)
    internal view returns (uint loanAmount, uint interestAmount) {
        // calculate loan values based on repayment amount
        loanAmount = repaymentAmount.mul(product.discountRate).div(1000000);
        interestAmount = loanAmount > repaymentAmount ? 0 : repaymentAmount.sub(loanAmount);
    }

    /* internal function, assuming repayment amount already transfered  */
    function _repayLoan(uint loanId, uint repaymentAmount) internal {
        require(loanId < loans.length, "invalid loanId"); // next line would revert but require to emit reason
        LoanData storage loan = loans[loanId];
        require(loan.state == LoanState.Open, "loan state must be Open");
        require(repaymentAmount == loan.repaymentAmount, "repaymentAmount must be equal to tokens sent");
        require(now <= loan.maturity, "current time must be earlier than maturity");

        LoanProduct storage product = products[loan.productId];
        uint loanAmount;
        uint interestAmount;
        (loanAmount, interestAmount) = calculateLoanValues(product, loan.repaymentAmount);

        loans[loanId].state = LoanState.Repaid;

        if (interestAmount > 0) {
            augmintToken.transfer(monetarySupervisor.interestEarnedAccount(), interestAmount);
            augmintToken.burn(loanAmount);
        } else {
            // negative or zero interest (i.e. discountRate >= 0)
            augmintToken.burn(repaymentAmount);
        }

        monetarySupervisor.loanRepaymentNotification(loanAmount); // update KPIs

        loan.borrower.transfer(loan.collateralAmount); // send back ETH collateral

        emit LoanRepayed(loanId, loan.borrower);
    }
}

// File: contracts/Locker.sol

/* contract for tracking locked funds

 requirements
  -> lock funds
  -> unlock funds
  -> index locks by address

 For flows see: https://github.com/Augmint/augmint-contracts/blob/master/docs/lockFlow.png

 TODO / think about:
  -> self-destruct function?

*/

pragma solidity 0.4.24;







contract Locker is Restricted, TokenReceiver {

    using SafeMath for uint256;

    event NewLockProduct(uint32 indexed lockProductId, uint32 perTermInterest, uint32 durationInSecs,
                    uint32 minimumLockAmount, bool isActive);

    event LockProductActiveChange(uint32 indexed lockProductId, bool newActiveState);

    // NB: amountLocked includes the original amount, plus interest
    event NewLock(address indexed lockOwner, uint lockId, uint amountLocked, uint interestEarned,
                    uint40 lockedUntil, uint32 perTermInterest, uint32 durationInSecs);

    event LockReleased(address indexed lockOwner, uint lockId);

    event MonetarySupervisorChanged(MonetarySupervisor newMonetarySupervisor);

    struct LockProduct {
        // perTermInterest is in millionths (i.e. 1,000,000 = 100%):
        uint32 perTermInterest;
        uint32 durationInSecs;
        uint32 minimumLockAmount;
        bool isActive;
    }

    /* NB: we don't need to store lock parameters because lockProducts can't be altered (only disabled/enabled) */
    struct Lock {
        uint amountLocked;
        address owner;
        uint32 productId;
        uint40 lockedUntil;
        bool isActive;
    }

    AugmintTokenInterface public augmintToken;
    MonetarySupervisor public monetarySupervisor;

    LockProduct[] public lockProducts;

    Lock[] public locks;

    // lock ids for an account
    mapping(address => uint[]) public accountLocks;

    constructor(address permissionGranterContract, AugmintTokenInterface _augmintToken,
                    MonetarySupervisor _monetarySupervisor)
    public Restricted(permissionGranterContract) {
        augmintToken = _augmintToken;
        monetarySupervisor = _monetarySupervisor;

    }

    function addLockProduct(uint32 perTermInterest, uint32 durationInSecs, uint32 minimumLockAmount, bool isActive)
    external restrict("StabilityBoard") {
        uint _newLockProductId = lockProducts.push(
                                    LockProduct(perTermInterest, durationInSecs, minimumLockAmount, isActive)) - 1;
        uint32 newLockProductId = uint32(_newLockProductId);
        require(newLockProductId == _newLockProductId, "lockProduct overflow");
        emit NewLockProduct(newLockProductId, perTermInterest, durationInSecs, minimumLockAmount, isActive);
    }

    function setLockProductActiveState(uint32 lockProductId, bool isActive) external restrict("StabilityBoard") {
        // next line would revert but require to emit reason:
        require(lockProductId < lockProducts.length, "invalid lockProductId");

        lockProducts[lockProductId].isActive = isActive;
        emit LockProductActiveChange(lockProductId, isActive);
    }

    /* lock funds, called from AugmintToken's transferAndNotify
     Flow for locking tokens:
        1) user calls token contract's transferAndNotify lockProductId passed in data arg
        2) transferAndNotify transfers tokens to the Lock contract
        3) transferAndNotify calls Lock.transferNotification with lockProductId
    */
    function transferNotification(address from, uint256 amountToLock, uint _lockProductId) external {
        require(msg.sender == address(augmintToken), "msg.sender must be augmintToken");
        // next line would revert but require to emit reason:
        require(_lockProductId < lockProducts.length, "invalid lockProductId");
        uint32 lockProductId = uint32(_lockProductId);
        /* TODO: make data arg generic bytes
            uint productId;
            assembly { // solhint-disable-line no-inline-assembly
                productId := mload(data)
        } */
        _createLock(lockProductId, from, amountToLock);
    }

    function releaseFunds(uint lockId) external {
        // next line would revert but require to emit reason:
        require(lockId < locks.length, "invalid lockId");
        Lock storage lock = locks[lockId];
        LockProduct storage lockProduct = lockProducts[lock.productId];

        require(lock.isActive, "lock must be in active state");
        require(now >= lock.lockedUntil, "current time must be later than lockedUntil");

        lock.isActive = false;

        uint interestEarned = calculateInterest(lockProduct.perTermInterest, lock.amountLocked);

        monetarySupervisor.releaseFundsNotification(lock.amountLocked); // to maintain totalLockAmount
        augmintToken.transferWithNarrative(lock.owner, lock.amountLocked.add(interestEarned),
                                                                                "Funds released from lock");

        emit LockReleased(lock.owner, lockId);
    }

    function setMonetarySupervisor(MonetarySupervisor newMonetarySupervisor) external restrict("StabilityBoard") {
        monetarySupervisor = newMonetarySupervisor;
        emit MonetarySupervisorChanged(newMonetarySupervisor);
    }

    function getLockProductCount() external view returns (uint) {
        return lockProducts.length;
    }

    // returns <chunkSize> lock products starting from some <offset>
    // lock products are encoded as [ perTermInterest, durationInSecs, minimumLockAmount, maxLockAmount, isActive ]
    function getLockProducts(uint offset, uint16 chunkSize)
    external view returns (uint[5][]) {
        uint limit = SafeMath.min(offset.add(chunkSize), lockProducts.length);
        uint[5][] memory response = new uint[5][](limit.sub(offset));

        for (uint i = offset; i < limit; i++) {
            LockProduct storage lockProduct = lockProducts[i];
            response[i - offset] = [lockProduct.perTermInterest, lockProduct.durationInSecs, lockProduct.minimumLockAmount,
                        monetarySupervisor.getMaxLockAmount(lockProduct.minimumLockAmount, lockProduct.perTermInterest),
                        lockProduct.isActive ? 1 : 0 ];
        }
        return response;
    }

    function getLockCount() external view returns (uint) {
        return locks.length;
    }

    function getLockCountForAddress(address lockOwner) external view returns (uint) {
        return accountLocks[lockOwner].length;
    }

    // returns <chunkSize> locks starting from some <offset>
    // lock products are encoded as
    //       [lockId, owner, amountLocked, interestEarned, lockedUntil, perTermInterest, durationInSecs, isActive ]
    // NB: perTermInterest is in millionths (i.e. 1,000,000 = 100%):
    function getLocks(uint offset, uint16 chunkSize)
    external view returns (uint[8][]) {
        uint limit = SafeMath.min(offset.add(chunkSize), locks.length);
        uint[8][] memory response = new uint[8][](limit.sub(offset));

        for (uint i = offset; i < limit; i++) {
            Lock storage lock = locks[i];
            LockProduct storage lockProduct = lockProducts[lock.productId];
            uint interestEarned = calculateInterest(lockProduct.perTermInterest, lock.amountLocked);

            response[i - offset] = [uint(i), uint(lock.owner), lock.amountLocked, interestEarned, lock.lockedUntil,
                        lockProduct.perTermInterest, lockProduct.durationInSecs, lock.isActive ? 1 : 0];
        }
        return response;
    }

    // returns <chunkSize> locks of a given account, starting from some <offset>
    // lock products are encoded as
    //             [lockId, amountLocked, interestEarned, lockedUntil, perTermInterest, durationInSecs, isActive ]
    function getLocksForAddress(address lockOwner, uint offset, uint16 chunkSize)
    external view returns (uint[7][]) {
        uint[] storage locksForAddress = accountLocks[lockOwner];
        uint limit = SafeMath.min(offset.add(chunkSize), locksForAddress.length);
        uint[7][] memory response = new uint[7][](limit.sub(offset));

        for (uint i = offset; i < limit; i++) {
            Lock storage lock = locks[locksForAddress[i]];
            LockProduct storage lockProduct = lockProducts[lock.productId];

            uint interestEarned = calculateInterest(lockProduct.perTermInterest, lock.amountLocked);

            response[i - offset] = [locksForAddress[i], lock.amountLocked, interestEarned, lock.lockedUntil,
                        lockProduct.perTermInterest, lockProduct.durationInSecs, lock.isActive ? 1 : 0 ];
        }
        return response;
    }

    function calculateInterest(uint32 perTermInterest, uint amountToLock) public pure returns (uint interestEarned) {
        interestEarned = amountToLock.mul(perTermInterest).ceilDiv(1000000);
    }

    // Internal function. assumes amountToLock is already transferred to this Lock contract
    function _createLock(uint32 lockProductId, address lockOwner, uint amountToLock) internal {
        LockProduct storage lockProduct = lockProducts[lockProductId];
        require(lockProduct.isActive, "lockProduct must be in active state");
        require(amountToLock >= lockProduct.minimumLockAmount, "amountToLock must be >= minimumLockAmount");

        uint interestEarned = calculateInterest(lockProduct.perTermInterest, amountToLock);
        uint expiration = now.add(lockProduct.durationInSecs);
        uint40 lockedUntil = uint40(expiration);
        require(lockedUntil == expiration, "lockedUntil overflow");

        uint lockId = locks.push(Lock(amountToLock, lockOwner, lockProductId, lockedUntil, true)) - 1;
        accountLocks[lockOwner].push(lockId);

        monetarySupervisor.requestInterest(amountToLock, interestEarned); // update KPIs & transfer interest here

        emit NewLock(lockOwner, lockId, amountToLock, interestEarned, lockedUntil, lockProduct.perTermInterest,
                    lockProduct.durationInSecs);
    }
}

// File: contracts/PreToken.sol

/* Augmint pretoken contract to record agreements and tokens allocated based on the agreement.

    Important: this is NOT an ERC20 token!

    PreTokens are non-fungible: agreements can have different conditions (valuationCap and discount)
        and pretokens are not tradable.

    Ownership can be transferred if owner wants to change wallet but the whole agreement and
        the total pretoken amount is moved to a new account

    PreTokenSigner can (via MultiSig):
      - add agreements and issue pretokens to an agreement
      - change owner of any agreement to handle if an owner lost a private keys
      - burn pretokens from any agreement to fix potential erroneous issuance
    These are known compromises on trustlessness hence all these tokens distributed based on signed agreements and
        preTokens are issued only to a closed group of contributors / team members.
    If despite these something goes wrong then as a last resort a new pretoken contract can be recreated from agreements.

    Some ERC20 functions are implemented so agreement owners can see their balances and use transfer in standard wallets.
    Restrictions:
      - only total balance can be transfered - effectively ERC20 transfer used to transfer agreement ownership
      - only agreement holders can transfer
        (i.e. can't transfer 0 amount if have no agreement to avoid polluting logs with Transfer events)
      - transfer is only allowed to accounts without an agreement yet
      - no approval and transferFrom ERC20 functions
 */

pragma solidity 0.4.24;




contract PreToken is Restricted {
    using SafeMath for uint256;

    string constant public name = "Augmint pretokens"; // solhint-disable-line const-name-snakecase
    string constant public symbol = "APRE"; // solhint-disable-line const-name-snakecase
    uint8 constant public decimals = 0; // solhint-disable-line const-name-snakecase

    uint public totalSupply;

    struct Agreement {
        address owner;
        uint balance;
        uint32 discount; //  discountRate in parts per million , ie. 10,000 = 1%
        uint32 valuationCap; // in USD (no decimals)
    }

    /* Agreement hash is the SHA-2 (SHA-256) hash of signed agreement document.
         To generate:
            OSX: shasum -a 256 agreement.pdf
            Windows: certUtil -hashfile agreement.pdf SHA256 */
    mapping(address => bytes32) public agreementOwners; // to lookup agrement by owner
    mapping(bytes32 => Agreement) public agreements;

    bytes32[] public allAgreements; // all agreements to able to iterate over

    event Transfer(address indexed from, address indexed to, uint amount);

    event NewAgreement(address owner, bytes32 agreementHash, uint32 discount, uint32 valuationCap);

    constructor(address permissionGranterContract)
    public Restricted(permissionGranterContract) {} // solhint-disable-line no-empty-blocks

    function addAgreement(address owner, bytes32 agreementHash, uint32 discount, uint32 valuationCap)
    external restrict("PreTokenSigner") {
        require(owner != address(0), "owner must not be 0x0");
        require(agreementOwners[owner] == 0x0, "owner must not have an aggrement yet");
        require(agreementHash != 0x0, "agreementHash must not be 0x0");
        require(discount > 0, "discount must be > 0");
        require(agreements[agreementHash].discount == 0, "agreement must not exist yet");

        agreements[agreementHash] = Agreement(owner, 0, discount, valuationCap);
        agreementOwners[owner] = agreementHash;
        allAgreements.push(agreementHash);

        emit NewAgreement(owner, agreementHash, discount, valuationCap);
    }

    function issueTo(bytes32 agreementHash, uint amount) external restrict("PreTokenSigner") {
        Agreement storage agreement = agreements[agreementHash];
        require(agreement.discount > 0, "agreement must exist");

        agreement.balance = agreement.balance.add(amount);
        totalSupply = totalSupply.add(amount);

        emit Transfer(0x0, agreement.owner, amount);
    }

    /* Restricted function to allow pretoken signers to fix incorrect issuance */
    function burnFrom(bytes32 agreementHash, uint amount)
    public restrict("PreTokenSigner") returns (bool) {
        Agreement storage agreement = agreements[agreementHash];
        // this is redundant b/c of next requires but be explicit
        require(agreement.discount > 0, "agreement must exist");
        require(amount > 0, "burn amount must be > 0");
        // .sub would revert anyways but emit reason
        require(agreement.balance >= amount, "must not burn more than balance");

        agreement.balance = agreement.balance.sub(amount);
        totalSupply = totalSupply.sub(amount);

        emit Transfer(agreement.owner, 0x0, amount);
        return true;
    }

    function balanceOf(address owner) public view returns (uint) {
        return agreements[agreementOwners[owner]].balance;
    }

    /* function to transfer agreement ownership to other wallet by owner
        it's in ERC20 form so owners can use standard ERC20 wallet just need to pass full balance as value */
    function transfer(address to, uint amount) public returns (bool) { // solhint-disable-line no-simple-event-func-name
        require(amount == agreements[agreementOwners[msg.sender]].balance, "must transfer full balance");
        _transfer(msg.sender, to);
        return true;
    }

    /* Restricted function to allow pretoken signers to fix if pretoken owner lost keys */
    function transferAgreement(bytes32 agreementHash, address to)
    public restrict("PreTokenSigner") returns (bool) {
        _transfer(agreements[agreementHash].owner, to);
        return true;
    }

    /* private function used by transferAgreement & transfer */
    function _transfer(address from, address to) private {
        Agreement storage agreement = agreements[agreementOwners[from]];
        require(agreementOwners[from] != 0x0, "from agreement must exists");
        require(agreementOwners[to] == 0, "to must not have an agreement");
        require(to != 0x0, "must not transfer to 0x0");

        agreement.owner = to;

        agreementOwners[to] = agreementOwners[from];
        agreementOwners[from] = 0x0;

        emit Transfer(from, to, agreement.balance);
    }

    function getAgreementsCount() external view returns (uint agreementsCount) {
        return allAgreements.length;
    }

    // UI helper fx - Returns <chunkSize> agreements from <offset> as
    // [index in allAgreements, account address as uint, balance, agreementHash as uint,
    //          discount as uint, valuationCap as uint ]
    function getAgreements(uint offset, uint16 chunkSize)
    external view returns(uint[6][]) {
        uint limit = SafeMath.min(offset.add(chunkSize), allAgreements.length);
        uint[6][] memory response = new uint[6][](limit.sub(offset));

        for (uint i = offset; i < limit; i++) {
            bytes32 agreementHash = allAgreements[i];
            Agreement storage agreement = agreements[agreementHash];

            response[i - offset] = [i, uint(agreement.owner), agreement.balance,
                uint(agreementHash), uint(agreement.discount), uint(agreement.valuationCap)];
        }
        return response;
    }
}

// File: contracts/generic/MultiSig.sol

/* Abstract multisig contract to allow multi approval execution of atomic contracts scripts
        e.g. migrations or settings.
    * Script added by signing a script address by a signer  (NEW state)
    * Script goes to ALLOWED state once a quorom of signers sign it (quorom fx is defined in each derived contracts)
    * Script can be signed even in APPROVED state
    * APPROVED scripts can be executed only once.
        - if script succeeds then state set to DONE
        - If script runs out of gas or reverts then script state set to FAILEd and not allowed to run again
          (To avoid leaving "behind" scripts which fail in a given state but eventually execute in the future)
    * Scripts can be cancelled by an other multisig script approved and calling cancelScript()
    * Adding/removing signers is only via multisig approved scripts using addSigners / removeSigners fxs
*/
pragma solidity 0.4.24;



contract MultiSig {
    using SafeMath for uint256;

    mapping(address => bool) public isSigner;
    address[] public allSigners; // all signers, even the disabled ones
                                // NB: it can contain duplicates when a signer is added, removed then readded again
                                //   the purpose of this array is to being able to iterate on signers in isSigner
    uint public activeSignersCount;

    enum ScriptState {New, Approved, Done, Cancelled, Failed}

    struct Script {
        ScriptState state;
        uint signCount;
        mapping(address => bool) signedBy;
        address[] allSigners;
    }

    mapping(address => Script) public scripts;
    address[] public scriptAddresses;

    event SignerAdded(address signer);
    event SignerRemoved(address signer);

    event ScriptSigned(address scriptAddress, address signer);
    event ScriptApproved(address scriptAddress);
    event ScriptCancelled(address scriptAddress);

    event ScriptExecuted(address scriptAddress, bool result);

    constructor() public {
        // deployer address is the first signer. Deployer can configure new contracts by itself being the only "signer"
        // The first script which sets the new contracts live should add signers and revoke deployer's signature right
        isSigner[msg.sender] = true;
        allSigners.push(msg.sender);
        activeSignersCount = 1;
        emit SignerAdded(msg.sender);
    }

    function sign(address scriptAddress) public {
        require(isSigner[msg.sender], "sender must be signer");
        Script storage script = scripts[scriptAddress];
        require(script.state == ScriptState.Approved || script.state == ScriptState.New,
                "script state must be New or Approved");
        require(!script.signedBy[msg.sender], "script must not be signed by signer yet");

        if (script.allSigners.length == 0) {
            // first sign of a new script
            scriptAddresses.push(scriptAddress);
        }

        script.allSigners.push(msg.sender);
        script.signedBy[msg.sender] = true;
        script.signCount = script.signCount.add(1);

        emit ScriptSigned(scriptAddress, msg.sender);

        if (checkQuorum(script.signCount)) {
            script.state = ScriptState.Approved;
            emit ScriptApproved(scriptAddress);
        }
    }

    function execute(address scriptAddress) public returns (bool result) {
        // only allow execute to signers to avoid someone set an approved script failed by calling it with low gaslimit
        require(isSigner[msg.sender], "sender must be signer");
        Script storage script = scripts[scriptAddress];
        require(script.state == ScriptState.Approved, "script state must be Approved");

        // passing scriptAddress to allow called script access its own public fx-s if needed
        if (scriptAddress.delegatecall.gas(gasleft() - 23000)
            (abi.encodeWithSignature("execute(address)", scriptAddress))) {
            script.state = ScriptState.Done;
            result = true;
        } else {
            script.state = ScriptState.Failed;
            result = false;
        }
        emit ScriptExecuted(scriptAddress, result);
    }

    function cancelScript(address scriptAddress) public {
        require(msg.sender == address(this), "only callable via MultiSig");
        Script storage script = scripts[scriptAddress];
        require(script.state == ScriptState.Approved || script.state == ScriptState.New,
                "script state must be New or Approved");

        script.state = ScriptState.Cancelled;

        emit ScriptCancelled(scriptAddress);
    }

    /* requires quorum so it's callable only via a script executed by this contract */
    function addSigners(address[] signers) public {
        require(msg.sender == address(this), "only callable via MultiSig");
        for (uint i= 0; i < signers.length; i++) {
            if (!isSigner[signers[i]]) {
                require(signers[i] != address(0), "new signer must not be 0x0");
                activeSignersCount++;
                allSigners.push(signers[i]);
                isSigner[signers[i]] = true;
                emit SignerAdded(signers[i]);
            }
        }
    }

    /* requires quorum so it's callable only via a script executed by this contract */
    function removeSigners(address[] signers) public {
        require(msg.sender == address(this), "only callable via MultiSig");
        for (uint i= 0; i < signers.length; i++) {
            if (isSigner[signers[i]]) {
                require(activeSignersCount > 1, "must not remove last signer");
                activeSignersCount--;
                isSigner[signers[i]] = false;
                emit SignerRemoved(signers[i]);
            }
        }
    }

    /* implement it in derived contract */
    function checkQuorum(uint signersCount) internal view returns(bool isQuorum);

    function getAllSignersCount() view external returns (uint allSignersCount) {
        return allSigners.length;
    }

    // UI helper fx - Returns signers from offset as [signer id (index in allSigners), address as uint, isActive 0 or 1]
    function getSigners(uint offset, uint16 chunkSize)
    external view returns(uint[3][]) {
        uint limit = SafeMath.min(offset.add(chunkSize), allSigners.length);
        uint[3][] memory response = new uint[3][](limit.sub(offset));
        for (uint i = offset; i < limit; i++) {
            address signerAddress = allSigners[i];
            response[i - offset] = [i, uint(signerAddress), isSigner[signerAddress] ? 1 : 0];
        }
        return response;
    }

    function getScriptsCount() view external returns (uint scriptsCount) {
        return scriptAddresses.length;
    }

    // UI helper fx - Returns scripts from offset as
    //  [scriptId (index in scriptAddresses[]), address as uint, state, signCount]
    function getScripts(uint offset, uint16 chunkSize)
    external view returns(uint[4][]) {
        uint limit = SafeMath.min(offset.add(chunkSize), scriptAddresses.length);
        uint[4][] memory response = new uint[4][](limit.sub(offset));
        for (uint i = offset; i < limit; i++) {
            address scriptAddress = scriptAddresses[i];
            response[i - offset] = [i, uint(scriptAddress),
                uint(scripts[scriptAddress].state), scripts[scriptAddress].signCount];
        }
        return response;
    }
}

// File: contracts/PreTokenProxy.sol

/* allows tx to execute if 50% +1 vote of active signers signed */
pragma solidity 0.4.24;



contract PreTokenProxy is MultiSig {

    function checkQuorum(uint signersCount) internal view returns(bool isQuorum) {
        isQuorum = signersCount > activeSignersCount / 2 ;
    }
}

// File: contracts/StabilityBoardProxy.sol

/* allows tx to execute if 50% +1 vote of active signers signed */
pragma solidity 0.4.24;



contract StabilityBoardProxy is MultiSig {

    function checkQuorum(uint signersCount) internal view returns(bool isQuorum) {
        isQuorum = signersCount > activeSignersCount / 2 ;
    }
}

// File: contracts/TokenAEur.sol

/* Augmint Euro token (A-EUR) implementation */
pragma solidity 0.4.24;




contract TokenAEur is AugmintToken {
    constructor(address _permissionGranterContract, TransferFeeInterface _feeAccount)
    public AugmintToken(_permissionGranterContract, "Augmint Euro", "AEUR", "EUR", 2, _feeAccount)
    {} // solhint-disable-line no-empty-blocks

}

// File: contracts/SB_scripts/mainnet/Main0014_initNewContracts.sol

/* Set up new contracts */

pragma solidity 0.4.24;













contract Main0014_initNewContracts {

    /******************************************************************************
     * External dependencies
     ******************************************************************************/
    address public constant RATES_FEEDER_ACCOUNT = 0x8C58187a978979947b88824DCdA5Cb5fD4410387;

    /******************************************************************************
     * New contracts
     ******************************************************************************/
    StabilityBoardProxy public constant STABILITY_BOARD_PROXY = StabilityBoardProxy(0xde36a8773531406dCBefFdfd3C7b89fCed7A9F84);
    AugmintReserves public constant AUGMINT_RESERVES = AugmintReserves(0x65F30f8DD20c707C1938CcAd7416c7381E6eB9C8);
    Exchange public constant EXCHANGE = Exchange(0xc670FFBfa21c37481FB4EF2eA2249B9B78D2B073);
    FeeAccount public constant FEE_ACCOUNT = FeeAccount(0xE3ED84A163b9EeaF4f69B4890ae45cC52171Aa7E);
    InterestEarnedAccount public constant INTEREST_EARNED_ACCOUNT = InterestEarnedAccount(0xf23e0AF0e41341127Bb4e7b203aebCA0185f9EbD);
    LoanManager public constant LOAN_MANAGER = LoanManager(0x1cABc34618ecf2949F0405A86353e7705E01C38b);
    Locker public constant LOCKER = Locker(0x5Cc161482E82f20840A4AAEB582beCBCC4b539D7);
    MonetarySupervisor public constant MONETARY_SUPERVISOR = MonetarySupervisor(0x27484AFe9e6c332fB07F21Fac82d442EBe1D22c3);
    Rates public constant RATES = Rates(0x4272dB2EB82068E898588C3D6e4B5D55c3848793);
    TokenAEur public constant TOKEN_AEUR = TokenAEur(0xc994a2dEb02543Db1f48688438b9903c4b305ce3);
    PreToken public constant PRE_TOKEN = PreToken(0x97ea02179801FA94227DB5fC1d13Ac4277d40920);
    PreTokenProxy public constant PRE_TOKEN_PROXY = PreTokenProxy(0x8a69cf9d1D85bC150F69FeB80cC34c552F5fbea9);

    function execute(Main0014_initNewContracts /* self, not used */) external {
        // called via StabilityBoardProxy
        require(address(this) == address(STABILITY_BOARD_PROXY), "only execute via StabilityBoardProxy");

        /******************************************************************************
         * Set up permissions
         ******************************************************************************/

        // StabilityBoard permission
        AUGMINT_RESERVES.grantPermission(address(STABILITY_BOARD_PROXY), "StabilityBoard");
        EXCHANGE.grantPermission(address(STABILITY_BOARD_PROXY), "StabilityBoard");
        FEE_ACCOUNT.grantPermission(address(STABILITY_BOARD_PROXY), "StabilityBoard");
        INTEREST_EARNED_ACCOUNT.grantPermission(address(STABILITY_BOARD_PROXY), "StabilityBoard");
        LOAN_MANAGER.grantPermission(address(STABILITY_BOARD_PROXY), "StabilityBoard");
        LOCKER.grantPermission(address(STABILITY_BOARD_PROXY), "StabilityBoard");
        MONETARY_SUPERVISOR.grantPermission(address(STABILITY_BOARD_PROXY), "StabilityBoard");
        RATES.grantPermission(address(STABILITY_BOARD_PROXY), "StabilityBoard");
        TOKEN_AEUR.grantPermission(address(STABILITY_BOARD_PROXY), "StabilityBoard");

        // RatesFeeder permission (allows calling setRate)
        RATES.grantPermission(address(RATES_FEEDER_ACCOUNT), "RatesFeeder");

        // NoTransferFee permission
        FEE_ACCOUNT.grantPermission(address(AUGMINT_RESERVES), "NoTransferFee");
        FEE_ACCOUNT.grantPermission(address(EXCHANGE), "NoTransferFee");
        FEE_ACCOUNT.grantPermission(address(FEE_ACCOUNT), "NoTransferFee");
        FEE_ACCOUNT.grantPermission(address(INTEREST_EARNED_ACCOUNT), "NoTransferFee");
        FEE_ACCOUNT.grantPermission(address(LOAN_MANAGER), "NoTransferFee");
        FEE_ACCOUNT.grantPermission(address(LOCKER), "NoTransferFee");
        FEE_ACCOUNT.grantPermission(address(MONETARY_SUPERVISOR), "NoTransferFee");

        // MonetarySupervisor permission
        AUGMINT_RESERVES.grantPermission(address(MONETARY_SUPERVISOR), "MonetarySupervisor");
        INTEREST_EARNED_ACCOUNT.grantPermission(address(MONETARY_SUPERVISOR), "MonetarySupervisor");
        TOKEN_AEUR.grantPermission(address(MONETARY_SUPERVISOR), "MonetarySupervisor");

        // LoanManager permission
        MONETARY_SUPERVISOR.grantPermission(address(LOAN_MANAGER), "LoanManager");

        // Locker permission
        MONETARY_SUPERVISOR.grantPermission(address(LOCKER), "Locker");


        //  PreToken signers and permissions

        address[] memory preTokenSigners = new address[](3);
        preTokenSigners[0] = 0xd8203A652452906586F2E6cB6e31f6f7fed094D4;  // Sz.K.
        preTokenSigners[1] = 0xf9ea0E2857405C859bb8647ECB11f931D1259753;  // P.P.
        PRE_TOKEN_PROXY.addSigners(preTokenSigners);

        PRE_TOKEN.grantPermission(address(PRE_TOKEN_PROXY), "PreTokenSigner");
        PRE_TOKEN.grantPermission(address(PRE_TOKEN_PROXY), "PermissionGranter");
        PRE_TOKEN.revokePermission(address(STABILITY_BOARD_PROXY), "PermissionGranter");


        // StabilityBoard signers in new StabilityBoardProxy on Mainnet

        address[] memory stabilityProxySigners = new address[](2);
        stabilityProxySigners[0] = 0x53DBF6E8fe46307C7536eAbb0D90CADA3e732716;    // Sz.V.
        stabilityProxySigners[1] = 0xAE162e28575Ba898dc08D283f2Be10AE8b4114A2;    // Sz.K.
        stabilityProxySigners[2] = 0x9de3F6E7caCbb7e1c2489dFCe21abbB0ecEE6213;    // P.P.
        STABILITY_BOARD_PROXY.addSigners(stabilityProxySigners);


        /******************************************************************************
         * Add loan products
         ******************************************************************************/
        // term (in sec), discountRate, loanCoverageRatio, minDisbursedAmount (w/ 4 decimals), defaultingFeePt, isActive
        LOAN_MANAGER.addLoanProduct(365 days, 953288, 600000, 800, 100000, true); // 4.9% p.a.
        LOAN_MANAGER.addLoanProduct(180 days, 976405, 600000, 800, 100000, true); // 4.9% p.a.
        LOAN_MANAGER.addLoanProduct(90 days, 988062, 600000, 800, 100000, true); // 4.9% p.a.
        LOAN_MANAGER.addLoanProduct(30 days, 995988, 600000, 800, 100000, true); // 4.9% p.a.
        LOAN_MANAGER.addLoanProduct(14 days, 998124, 600000, 800, 100000, true); // 4.9% p.a.
        LOAN_MANAGER.addLoanProduct(7 days, 999062, 600000, 800, 100000, true); // 4.9% p.a.


        /******************************************************************************
         * Add lock products
         ******************************************************************************/
        // (perTermInterest, durationInSecs, minimumLockAmount, isActive)
        LOCKER.addLockProduct(45000, 365 days, 1000, true);  // 4.5% p.a.
        LOCKER.addLockProduct(22192, 180 days, 1000, true);  // 4.5% p.a.
        LOCKER.addLockProduct(11096, 90 days, 1000, true);  // 4.5% p.a.
        LOCKER.addLockProduct(3699, 30 days, 1000, true);  // 4.5% p.a.
        LOCKER.addLockProduct(1727, 14 days, 1000, true);  // 4.5% p.a.
        LOCKER.addLockProduct(864, 7 days, 1000, true);    // 4.5% p.a.

    }
}
