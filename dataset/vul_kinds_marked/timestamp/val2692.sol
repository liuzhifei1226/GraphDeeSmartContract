pragma solidity ^0.4.21;

contract ERC20Basic {
    function totalSupply() public view returns (uint256);
    function balanceOf(address who) public view returns (uint256);
    function transfer(address to, uint256 value) public returns (bool);
    event Transfer(address indexed from, address indexed to, uint256 value);
}
contract ERC20 is ERC20Basic {
    function allowance(address owner, address spender) public view returns (uint256);
    function transferFrom(address from, address to, uint256 value) public returns (bool);
    function approve(address spender, uint256 value) public returns (bool);
    event Approval(address indexed owner, address indexed spender, uint256 value);
}
library SafeERC20 {
    function safeTransfer(ERC20 token, address to, uint256 value) internal {
        assert(token.transfer(to, value));
    }
    function safeTransferFrom(ERC20 token, address from, address to, uint256 value) internal{
        assert(token.transferFrom(from, to, value));
    }
}

library SafeMath {
  function mul(uint256 a, uint256 b) internal pure returns (uint256) {
    uint256 c = a * b;
    assert(a == 0 || c / a == b);
    return c;
  }

  function div(uint256 a, uint256 b) internal pure returns (uint256) {
    uint256 c = a / b;
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

contract TOSMidHoldingContract is owned{
    using SafeERC20 for ERC20;
    using SafeMath for uint;
    string public constant name = "TOSMidHoldingContract";
    uint[6] public releasePercentages = [
        15,  //15%
        35,   //20%
        50,   //15%
        65,   //15%
        80,   //15%
        100   //20%
    ];

    uint256 public constant HOLDING_START               = 1533916800;  //2018/8/11 0:0:0
    uint256 public constant RELEASE_START               = 1541260800; //2018/11/4 0:0:0
    uint256 public constant RELEASE_INTERVAL            = 30 days; // 30 days
    uint256 public RELEASE_END                          = RELEASE_START.add(RELEASE_INTERVAL.mul(5));
    ERC20 public tosToken = ERC20(0xFb5a551374B656C6e39787B1D3A03fEAb7f3a98E);
    
    mapping (address => uint256) public lockBalanceOf;/// Locked account details
    mapping (address => uint256) public amountsRecords;
    mapping (address => uint256) public released;

    uint256 public totalLockAmount = 0; 
    function TOSMidHoldingContract() public {}
    function lock(uint256 lockAmount) public {

        require(lockAmount >= 100000 * 10 ** 18); /// > 100,000
        require(now <= HOLDING_START); 

        uint256 reward = lockAmount.mul(20).div(100);

        require(reward <= (tosToken.balanceOf(this).sub(totalLockAmount)));
        tosToken.safeTransferFrom(msg.sender, this, lockAmount);

        lockBalanceOf[msg.sender] = lockBalanceOf[msg.sender].add(lockAmount).add(reward);
        amountsRecords[msg.sender] = lockBalanceOf[msg.sender];
        totalLockAmount = totalLockAmount.add(lockAmount).add(reward);
    }

    function release() public {
        uint256 num = now.sub(RELEASE_START).div(RELEASE_INTERVAL);

        uint256 releaseAmount = 0;
        if (num >= releasePercentages.length.sub(1)) {
            releaseAmount = lockBalanceOf[msg.sender];
            released[msg.sender] = 100;
        }
        else {
            releaseAmount = amountsRecords[msg.sender].mul(releasePercentages[num].sub(released[msg.sender])).div(100);
            released[msg.sender] = releasePercentages[num];
        }

        require(releaseAmount > 0);
        tosToken.safeTransfer(msg.sender, releaseAmount);
        lockBalanceOf[msg.sender] = lockBalanceOf[msg.sender].sub(releaseAmount);
        totalLockAmount = totalLockAmount.sub(releaseAmount);
    }

    function remainingReward() public onlyOwner {
        require(now > HOLDING_START); 
        require(tosToken.balanceOf(this) > totalLockAmount);
        tosToken.safeTransfer(owner, tosToken.balanceOf(this).sub(totalLockAmount));
    }
}
