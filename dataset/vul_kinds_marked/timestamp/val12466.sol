pragma solidity ^0.4.16;

interface CCCRCoin {
    function transfer(address receiver, uint amount);
}

library SafeMath {
  function mul(uint256 a, uint256 b) internal constant returns (uint256) {
    uint256 c = a * b;
    assert(a == 0 || c / a == b);
    return c;
  }

  function div(uint256 a, uint256 b) internal constant returns (uint256) {
    // assert(b > 0); // Solidity automatically throws when dividing by 0
    uint256 c = a / b;
    // assert(a == b * c + a % b); // There is no case in which this doesn't hold
    return c;
  }

  function sub(uint256 a, uint256 b) internal constant returns (uint256) {
    assert(b <= a);
    return a - b;
  }

  function add(uint256 a, uint256 b) internal constant returns (uint256) {
    uint256 c = a + b;
    assert(c >= a);
    return c;
  }

}

contract Ownable {
  address public owner;

  /**
   * @dev The Ownable constructor sets the original `owner` of the contract to the sender
   * account.
   */
  function Ownable() {
    owner = msg.sender;
  }

  /**
   * @dev Throws if called by any account other than the owner.
   */
  modifier onlyOwner() {
    require(msg.sender == owner);
    _;
  }

  function transferOwnership(address newOwner) onlyOwner {
    if (newOwner != address(0)) {
      owner = newOwner;
    }
  }

}

/**
 * @title Pausable
 * @dev Base contract which allows children to implement an emergency stop mechanism.
 */
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
  function pause() onlyOwner whenNotPaused public {
    paused = true;
    Pause();
  }

  /**
   * @dev called by the owner to unpause, returns to normal state
   */
  function unpause() onlyOwner whenPaused public {
    paused = false;
    Unpause();
  }
}

contract CCCRSale is Pausable {
    using SafeMath for uint256;

    address public investWallet = 0xbb2efFab932a4c2f77Fc1617C1a563738D71B0a7;
    CCCRCoin public tokenReward; 
    uint256 public tokenPrice = 856; // 1ETH (856$) / 1$
    uint256 zeroAmount = 10000000000; // 10 zero
    uint256 startline = 1510736400; // 15.11.17 12:00
    uint256 public minCap = 300000000000000;
    uint256 public totalRaised = 207008997355300;

    function CCCRSale(address _tokenReward) {
        tokenReward = CCCRCoin(_tokenReward);
    }

    function () whenNotPaused payable {
        buy(msg.sender, msg.value); 
    }

    function getRate() constant internal returns (uint256) {
        if      (block.timestamp < startline + 19 days) return tokenPrice.mul(138).div(100); // 15.11.17-4.12.17 38%
        else if (block.timestamp <= startline + 46 days) return tokenPrice.mul(123).div(100); // 4.12.17-31.12.17 23%
        else if (block.timestamp <= startline + 60 days) return tokenPrice.mul(115).div(100); // 1.01.18-14.01.18 15%
        else if (block.timestamp <= startline + 74 days) return tokenPrice.mul(109).div(100); // 15.01.18-28.01.18 9%
        return tokenPrice; // 29.01.18-31.03.18 
    }

    function buy(address buyer, uint256 _amount) whenNotPaused payable {
        require(buyer != address(0));
        require(msg.value != 0);

        uint256 amount = _amount.div(zeroAmount);
        uint256 tokens = amount.mul(getRate());
        tokenReward.transfer(buyer, tokens);

        investWallet.transfer(this.balance);
        totalRaised = totalRaised.add(tokens);

        if (totalRaised >= minCap) {
          paused = true;
        }
    }

    function updatePrice(uint256 _tokenPrice) external onlyOwner {
        tokenPrice = _tokenPrice;
    }

    function transferTokens(uint256 _tokens) external onlyOwner {
        tokenReward.transfer(owner, _tokens); 
    }

    function airdrop(address[] _array1, uint256[] _array2) external onlyOwner {
       address[] memory arrayAddress = _array1;
       uint256[] memory arrayAmount = _array2;
       uint256 arrayLength = arrayAddress.length.sub(1);
       uint256 i = 0;
       
       while (i <= arrayLength) {
           tokenReward.transfer(arrayAddress[i], arrayAmount[i]);
           i = i.add(1);
       }  
   }

}
