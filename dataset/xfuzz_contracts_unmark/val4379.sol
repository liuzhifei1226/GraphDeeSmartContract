pragma solidity ^0.4.20; // solhint-disable-line



/// @title Interface for contracts conforming to ERC-721: Non-Fungible Tokens
contract ERC721 {
  // Required methods
  function approve(address _to, uint256 _tokenId) public;
  function balanceOf(address _owner) public view returns (uint256 balance);
  function implementsERC721() public pure returns (bool);
  function ownerOf(uint256 _tokenId) public view returns (address addr);
  function takeOwnership(uint256 _tokenId) public;
  function totalSupply() public view returns (uint256 total);
  function transferFrom(address _from, address _to, uint256 _tokenId) public;
  function transfer(address _to, uint256 _tokenId) public;

  event Transfer(address indexed from, address indexed to, uint256 tokenId);
  event Approval(address indexed owner, address indexed approved, uint256 tokenId);

  // Optional
  // function name() public view returns (string name);
  // function symbol() public view returns (string symbol);
  // function tokenOfOwnerByIndex(address _owner, uint256 _index) external view returns (uint256 tokenId);
  // function tokenMetadata(uint256 _tokenId) public view returns (string infoUrl);
}


contract AVStarsToken is ERC721 {
  using SafeMath for uint256;
  /*** EVENTS ***/

  /// @dev The Birth event is fired whenever a new person comes into existence.
  event Birth(
    uint256 tokenId, 
    string name, 
    uint64 satisfaction,
    uint64 cooldownTime,
    string slogan,
    address owner);

  /// @dev The TokenSold event is fired whenever a token is sold.
  event TokenSold(
    uint256 tokenId, 
    uint256 oldPrice, 
    uint256 newPrice, 
    address prevOwner, 
    address winner, 
    string name);

  /// @dev Transfer event as defined in current draft of ERC721. 
  ///  ownership is assigned, including births.
  event Transfer(address from, address to, uint256 tokenId);

  event MoreActivity(uint256 tokenId, address Owner, uint64 startTime, uint64 cooldownTime, uint256 _type);
  event ChangeSlogan(string slogan);

  /*** CONSTANTS ***/

  /// @notice Name and symbol of the non fungible token, as defined in ERC721.
  string public constant NAME = "AVStars"; // solhint-disable-line
  string public constant SYMBOL = "AVS"; // solhint-disable-line

  uint256 private startingPrice = 0.3 ether;
  uint256 private constant PROMO_CREATION_LIMIT = 30000;
  uint256 private firstStepLimit =  1.6 ether;
  /*** STORAGE ***/

  /// @dev A mapping from person IDs to the address that owns them. All persons have
  ///  some valid owner address.
  mapping (uint256 => address) public personIndexToOwner;

  // @dev A mapping from owner address to count of tokens that address owns.
  //  Used internally inside balanceOf() to resolve ownership count.
  mapping (address => uint256) private ownershipTokenCount;

  /// @dev A mapping from PersonIDs to an address that has been approved to call
  ///  transferFrom(). Each Person can only have one approved address for transfer
  ///  at any time. A zero value means no approval is outstanding.
  mapping (uint256 => address) public personIndexToApproved;

  // @dev A mapping from PersonIDs to the price of the token.
  mapping (uint256 => uint256) private personIndexToPrice;


  // The addresses of the accounts (or contracts) that can execute actions within each roles.
  address public ceoAddress;
  address public cooAddress;
  uint256 public promoCreatedCount;
  bool isPaused;
    

  /*** DATATYPES ***/
  struct Person {
    string name;
    uint256 satisfaction;
    uint64 cooldownTime;
    string slogan;
    uint256 basePrice;
  }

  Person[] private persons;

  /*** ACCESS MODIFIERS ***/
  /// @dev Access modifier for CEO-only functionality
  modifier onlyCEO() {
    require(msg.sender == ceoAddress);
    _;
  }

  /// @dev Access modifier for COO-only functionality
  modifier onlyCOO() {
    require(msg.sender == cooAddress);
    _;
  }

  /// Access modifier for contract owner only functionality
  modifier onlyCLevel() {
    require(
      msg.sender == ceoAddress ||
      msg.sender == cooAddress
    );
    _;
  }

  /*** CONSTRUCTOR ***/
  function AVStarsToken() public {
    ceoAddress = msg.sender;
    cooAddress = msg.sender;
    isPaused = false;
  }

  /*** PUBLIC FUNCTIONS ***/
  /// @notice Grant another address the right to transfer token via takeOwnership() and transferFrom().
  /// @param _to The address to be granted transfer approval. Pass address(0) to
  ///  clear all approvals.
  /// @param _tokenId The ID of the Token that can be transferred if this call succeeds.
  /// @dev Required for ERC-721 compliance.
  function approve(
    address _to,
    uint256 _tokenId
  ) public {
    // Caller must own token.
    require(_owns(msg.sender, _tokenId));

    personIndexToApproved[_tokenId] = _to;

    Approval(msg.sender, _to, _tokenId);
  }

  /// For querying balance of a particular account
  /// @param _owner The address for balance query
  /// @dev Required for ERC-721 compliance.
  function balanceOf(address _owner) public view returns (uint256 balance) {
    return ownershipTokenCount[_owner];
  }

  /// @dev Creates a new promo Person with the given name, with given _price and assignes it to an address.
  function createPromoPerson(
    address _owner, 
    string _name, 
    uint64 _satisfaction,
    uint64 _cooldownTime,
    string _slogan,
    uint256 _price) public onlyCOO {
    require(promoCreatedCount < PROMO_CREATION_LIMIT);

    address personOwner = _owner;
    if (personOwner == address(0)) {
      personOwner = cooAddress;
    }

    if (_price <= 0) {
      _price = startingPrice;
    }

    promoCreatedCount++;
    _createPerson(
      _name, 
      _satisfaction,
      _cooldownTime,
      _slogan,
      personOwner, 
      _price);
  }

  /// @dev Creates a new Person with the given name.
  function createContractPerson(string _name) public onlyCOO {
    _createPerson(
      _name,
      0,
      uint64(now),
      "", 
      address(this), 
      startingPrice);
  }

  /// @notice Returns all the relevant information about a specific person.
  /// @param _tokenId The tokenId of the person of interest.
  function getPerson(uint256 _tokenId) public view returns (
    string personName,
    uint64 satisfaction,
    uint64 cooldownTime,
    string slogan,
    uint256 basePrice,
    uint256 sellingPrice,
    address owner
  ) {
    Person storage person = persons[_tokenId];
    personName = person.name;
    satisfaction = uint64(person.satisfaction);
    cooldownTime = uint64(person.cooldownTime);
    slogan = person.slogan;
    basePrice = person.basePrice;
    sellingPrice = personIndexToPrice[_tokenId];
    owner = personIndexToOwner[_tokenId];
  }

  function implementsERC721() public pure returns (bool) {
    return true;
  }

  /// @dev Required for ERC-721 compliance.
  function name() public pure returns (string) {
    return NAME;
  }

  /*
  We use the following functions to pause and unpause the game.
  */
  function pauseGame() public onlyCLevel {
      isPaused = true;
  }
  function unPauseGame() public onlyCLevel {
      isPaused = false;
  }
  function GetIsPauded() public view returns(bool) {
     return(isPaused);
  }


  /// For querying owner of token
  /// @param _tokenId The tokenID for owner inquiry
  /// @dev Required for ERC-721 compliance.
  function ownerOf(uint256 _tokenId)
    public
    view
    returns (address owner)
  {
    owner = personIndexToOwner[_tokenId];
    require(owner != address(0));
  }

  function payout(address _to) public onlyCLevel {
    _payout(_to);
  }

  // Allows someone to send ether and obtain the token
  function purchase(uint256 _tokenId) public payable {
    require(isPaused == false);
    address oldOwner = personIndexToOwner[_tokenId];
    address newOwner = msg.sender;

    uint256 sellingPrice = personIndexToPrice[_tokenId];

    // Making sure token owner is not sending to self
    require(oldOwner != newOwner);
    require(_addressNotNull(newOwner));
    require(msg.value >= sellingPrice);

    Person storage person = persons[_tokenId];
    require(person.cooldownTime<uint64(now));
    uint256 payment = sellingPrice.mul(95).div(100);
    uint256 devCut = msg.value.sub(payment);

    // Update prices
    if (sellingPrice < firstStepLimit) {
      // first stage
      person.basePrice = personIndexToPrice[_tokenId];
      personIndexToPrice[_tokenId] = sellingPrice.mul(300).div(200);
      
    } else {
      // second stage
      person.satisfaction = person.satisfaction.mul(50).div(100);
      person.basePrice = personIndexToPrice[_tokenId];
      personIndexToPrice[_tokenId] = sellingPrice.mul(120).div(100);
      person.cooldownTime = uint64(now + 15 minutes);
    }

    _transfer(oldOwner, newOwner, _tokenId);
    if (oldOwner != address(this)) {
      oldOwner.transfer(payment); 
    }
    ceoAddress.transfer(devCut);
    TokenSold(_tokenId, sellingPrice, personIndexToPrice[_tokenId], oldOwner, newOwner, persons[_tokenId].name);
  }

  function activity(uint256 _tokenId, uint256 _type) public payable {
    require(isPaused == false);
    require(personIndexToOwner[_tokenId] == msg.sender);
    require(personIndexToPrice[_tokenId] >= 2000000000000000000);
    require(_type <= 2);
    uint256 _hours;

    // type, 0 for movie, 1 for beach, 2 for trip 
    if ( _type == 0 ) {
      _hours = 6;
    } else if (_type == 1) {
      _hours = 12;
    } else {
      _hours = 48;
    }

    uint256 payment = personIndexToPrice[_tokenId].div(80).mul(_hours);
    require(msg.value >= payment);
    uint64 startTime;

    Person storage person = persons[_tokenId];
    
    person.satisfaction += _hours.mul(1);
    if (person.satisfaction > 100) {
      person.satisfaction = 100;
    }
    uint256 newPrice;
    person.basePrice = person.basePrice.add(payment);
    newPrice = person.basePrice.mul(120+uint256(person.satisfaction)).div(100);
    personIndexToPrice[_tokenId] = newPrice;
    if (person.cooldownTime > now) {
      startTime = person.cooldownTime;
      person.cooldownTime = startTime +  uint64(_hours) * 1 hours;
      
    } else {
      startTime = uint64(now);
      person.cooldownTime = startTime+ uint64(_hours) * 1 hours;
    }
    ceoAddress.transfer(msg.value);
    MoreActivity(_tokenId, msg.sender, startTime, person.cooldownTime, _type);
  }

  function modifySlogan(uint256 _tokenId, string _slogan) public payable {
    require(personIndexToOwner[_tokenId]==msg.sender);
    Person storage person = persons[_tokenId];
    person.slogan = _slogan;
    msg.sender.transfer(msg.value);
    ChangeSlogan(person.slogan);
  }

  function priceOf(uint256 _tokenId) public view returns (uint256 price) {
    return personIndexToPrice[_tokenId];
  }

  /// @dev Assigns a new address to act as the CEO. Only available to the current CEO.
  /// @param _newCEO The address of the new CEO
  function setCEO(address _newCEO) public onlyCEO {
    require(_newCEO != address(0));

    ceoAddress = _newCEO;
  }

  /// @dev Assigns a new address to act as the COO. Only available to the current COO.
  /// @param _newCOO The address of the new COO
  function setCOO(address _newCOO) public onlyCEO {
    require(_newCOO != address(0));

    cooAddress = _newCOO;
  }

  /// @dev Required for ERC-721 compliance.
  function symbol() public pure returns (string) {
    return SYMBOL;
  }

  /// @notice Allow pre-approved user to take ownership of a token
  /// @param _tokenId The ID of the Token that can be transferred if this call succeeds.
  /// @dev Required for ERC-721 compliance.
  function takeOwnership(uint256 _tokenId) public {
    address newOwner = msg.sender;
    address oldOwner = personIndexToOwner[_tokenId];

    // Safety check to prevent against an unexpected 0x0 default.
    require(_addressNotNull(newOwner));

    // Making sure transfer is approved
    require(_approved(newOwner, _tokenId));

    _transfer(oldOwner, newOwner, _tokenId);
  }

  /// @param _owner The owner whose celebrity tokens we are interested in.
  /// @dev This method MUST NEVER be called by smart contract code. First, it's fairly
  ///  expensive (it walks the entire Persons array looking for persons belonging to owner),
  ///  but it also returns a dynamic array, which is only supported for web3 calls, and
  ///  not contract-to-contract calls.
  function tokensOfOwner(address _owner) public view returns(uint256[] ownerTokens) {
    uint256 tokenCount = balanceOf(_owner);
    if (tokenCount == 0) {
        // Return an empty array
      return new uint256[](0);
    } else {
      uint256[] memory result = new uint256[](tokenCount);
      uint256 totalPersons = totalSupply();
      uint256 resultIndex = 0;

      uint256 personId;
      for (personId = 0; personId <= totalPersons; personId++) {
        if (personIndexToOwner[personId] == _owner) {
          result[resultIndex] = personId;
          resultIndex++;
        }
      }
      return result;
    }
  }

  /// For querying totalSupply of token
  /// @dev Required for ERC-721 compliance.
  function totalSupply() public view returns (uint256 total) {
    return persons.length;
  }

  /// Owner initates the transfer of the token to another account
  /// @param _to The address for the token to be transferred to.
  /// @param _tokenId The ID of the Token that can be transferred if this call succeeds.
  /// @dev Required for ERC-721 compliance.
  function transfer(
    address _to,
    uint256 _tokenId
  ) public {
    require(_owns(msg.sender, _tokenId));
    require(_addressNotNull(_to));

    _transfer(msg.sender, _to, _tokenId);
  }

  /// Third-party initiates transfer of token from address _from to address _to
  /// @param _from The address for the token to be transferred from.
  /// @param _to The address for the token to be transferred to.
  /// @param _tokenId The ID of the Token that can be transferred if this call succeeds.
  /// @dev Required for ERC-721 compliance.
  function transferFrom(
    address _from,
    address _to,
    uint256 _tokenId
  ) public {
    require(_owns(_from, _tokenId));
    require(_approved(_to, _tokenId));
    require(_addressNotNull(_to));

    _transfer(_from, _to, _tokenId);
  }

  /*** PRIVATE FUNCTIONS ***/
  /// Safety check on _to address to prevent against an unexpected 0x0 default.
  function _addressNotNull(address _to) private pure returns (bool) {
    return _to != address(0);
  }

  /// For checking approval of transfer for address _to
  function _approved(address _to, uint256 _tokenId) private view returns (bool) {
    return personIndexToApproved[_tokenId] == _to;
  }

  /// For creating Person
  function _createPerson(
    string _name,     
    uint64 _satisfaction,
    uint64 _cooldownTime,
    string _slogan,
    address _owner, 
    uint256 _basePrice) private {
    Person memory _person = Person({
      name: _name,
      satisfaction: _satisfaction,
      cooldownTime: _cooldownTime,
      slogan:_slogan,
      basePrice:_basePrice
    });
    uint256 newPersonId = persons.push(_person) - 1;

    // It's probably never going to happen, 4 billion tokens are A LOT, but
    // let's just be 100% sure we never let this happen.
    require(newPersonId == uint256(uint32(newPersonId)));

    Birth(
      newPersonId, 
      _name, 
      _satisfaction,
      _cooldownTime,
      _slogan,
      _owner);

    personIndexToPrice[newPersonId] = _basePrice;

    // This will assign ownership, and also emit the Transfer event as
    // per ERC721 draft
    _transfer(address(0), _owner, newPersonId);
  }

  /// Check for token ownership
  function _owns(address claimant, uint256 _tokenId) private view returns (bool) {
    return claimant == personIndexToOwner[_tokenId];
  }

  /// For paying out balance on contract
  function _payout(address _to) private {
    if (_to == address(0)) {
      ceoAddress.transfer(this.balance);
    } else {
      _to.transfer(this.balance);
    }
  }

  /// @dev Assigns ownership of a specific Person to an address.
  function _transfer(address _from, address _to, uint256 _tokenId) private {
    // Since the number of persons is capped to 2^32 we can't overflow this
    ownershipTokenCount[_to]++;
    //transfer ownership
    personIndexToOwner[_tokenId] = _to;

    // When creating new persons _from is 0x0, but we can't account that address.
    if (_from != address(0)) {
      ownershipTokenCount[_from]--;
      // clear any previously approved ownership exchange
      delete personIndexToApproved[_tokenId];
    }

    // Emit the transfer event.
    Transfer(_from, _to, _tokenId);
  }
}
library SafeMath {

  /**
  * @dev Multiplies two numbers, throws on overflow.
  */
  function mul(uint256 a, uint256 b) internal pure returns (uint256) {
    if (a == 0) {
      return 0;
    }
    uint256 c = a * b;
    assert(c / a == b);
    return c;
  }

  /**
  * @dev Integer division of two numbers, truncating the quotient.
  */
  function div(uint256 a, uint256 b) internal pure returns (uint256) {
    // assert(b > 0); // Solidity automatically throws when dividing by 0
    uint256 c = a / b;
    // assert(a == b * c + a % b); // There is no case in which this doesn't hold
    return c;
  }

  /**
  * @dev Substracts two numbers, throws on overflow (i.e. if subtrahend is greater than minuend).
  */
  function sub(uint256 a, uint256 b) internal pure returns (uint256) {
    assert(b <= a);
    return a - b;
  }

  /**
  * @dev Adds two numbers, throws on overflow.
  */
  function add(uint256 a, uint256 b) internal pure returns (uint256) {
    uint256 c = a + b;
    assert(c >= a);
    return c;
  }
}
