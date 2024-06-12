pragma solidity ^0.4.16;


/**
 * @title Ownable
 * @dev The Ownable contract has an owner address, and provides basic authorization control
 * functions, this simplifies the implementation of "user permissions".
 */
contract Ownable {
  address public owner;


  /**
   * @dev The Ownable constructor sets the original `owner` of the contract to the sender
   * account.
   */
  function Ownable() public {
    owner = msg.sender;
  }


  /**
   * @dev Throws if called by any account other than the owner.
   */
  modifier onlyOwner() {
    require(msg.sender == owner);
    _;
  }
}

interface MintableObject {
  function transfer(address _to, uint256 _value) external returns (bool);
  function transferOwnership(address newOwner) external;
  function mint(address _to, uint256 _amount) external returns (bool);
}

contract Minter is Ownable {
  MintableObject public token = MintableObject(0x02585E4A14dA274D02dF09b222D4606B10a4E940); // HeroOrigenToken contract address
  uint256 public index = 0;
  bool public complete = false;
  address[] public holders =   [
    0xe875e4C5EC4fF7e0f002Bb9F794C6464551A5b3a,
    0x4F5695d01cB2f9c1F8578A3Aa351eE38d537cbe3,
    0xa76977132E3E876c77b1BD8c673dACbA60d9e7e0,
    0x70203709D6A7863fa4F335F7135109C35fD4c19b,
    0xc18a235B1a80F9d6A60F963826dD60A582898D28,
    0xCC677048A3698FF431FD2856D6e91DF29b6183Cf,
    0xAf6D62e26Baa00889e18a46Fc8e2687d45764427,
    0x731Ec1abd1366965D321e01Df2AdDc634FAA3518,
    0x89D7601aB2f543e0417C956d90908BFeC51C03dA,
    0x9143e0B9Ed5bF0d95aeb60cfa507151036029A41,
    0x8c64d5B26C4ed5f44Ac7000db1e8031e8dDb6482,
    0x2b789589A025d7F0DcE4700E6d68366538Cf2c76,
    0xB9d6B8Bb8528CB14f1DD6BcC28014226d08c1970,
    0x9d1Fc06b6Da636922cFA676d00737a5A847Ff997,
    0x70203709D6A7863fa4F335F7135109C35fD4c19b,
    0x9E966A66021bA3838Ed3E1d9049D0E03ebD01130,
    0x9eDBABAA6Fc3F4C31116cc810730EF646A461312,
    0x77FD9368886f69eBB16fAdc7F7E7c62e7e9b9Eb2,
    0x7A434e1A3348d3A121d155D0935867d45BD699B0,
    0x98AEE42201B361df981425F9917EAD8375E4569f,
    0xbcaBD5CDBF1021A61a1E65e360ADf765375404BD,
    0xda7b0ee914C7FB7A5F45b4598379636e86CA8725,
    0x9405dAaebeb2e07cEe7Cb8BD21cB1f438a7D2Aa8,
    0x2ae256c42c4090eD0769A534EF1CE64D7299246F,
    0x8b16F2026dcCe8Cf0d1365FcbfD91e959692f2C2,
    0x62496C0bf7e253a80AFBfe41463C47cBff642602,
    0xB1366F8f379D3C3B8E1EB4B070BE0006b92fCBCb,
    0xBA6fE917B0951924350756BFC33f4412fcD8A84b,
    0xcD1fC1e8Ca1D07075e939918c51086749F6EA4C8,
    0x887f2Bd93fbF3792052C2c67e8c6B1106aB712f0,
    0xcBA7f4f44473e32Dd01cD863622b2B8797681955,
    0xF5fE4c4d0133a8fA85d9fD230449416717985582,
    0xA3cdF9a69A8743ee7cD43Ee5f766E147ee4f592e,
    0xDfCb03a82f2eE88Fd9C5B5a904762bc1c93F4A7d,
    0xa7b66707318a888F03356C2b809a687560Df6333,
    0xd7f75Cc0e1B6bcd713466D57F30b8176BAE38C9F,
    0x7698DC4aE9dCC82d302aD1336873f99914b428E7,
    0x4bD33CC493dC30B06ee741470D454eDc0975dca1,
    0xA488d81060f5663b079BE53f5810c6DA6d9584A9,
    0xA0E8937E700606551121D7f1AfCA67e85c1E200d,
    0x65d55520F74e627549BA9140c6811a76f18567Be,
    0x9cc214e167980171525a3cdacf852cb16283dbfa,
    0xce562CEED7e5c1AD4E665046bae4f51E7815cc55,
    0xdc96F5228D5659EEA2826234420014c91ca1De21,
    0x11AA7f62Eb80Ad7eEFe39b5139cf247bb278AD9C,
    0xDBf96CaC84EFcC4979557dA4cCCBeC69Bf483e02,
    0x12aB58DD2d0218E163AB40b5F551C9d51534F443,
    0xbaf7139e9Cf7aBa4fA0417a9E16E8EeBf9749B24,
    0x3e81a0844332bc69cae58c47d4ae881d978fa8f7,
    0x0Ccfe098090b9303854C8dC67804B8BFE21Cc4a0,
    0x107a99f040da5B62F3033d69C07Efc94956885F7,
    0x2e006284072fa77142cbed0caa41cdd646ecc381,
    0xf1ABa2E6140fEc6ccb178C7B97556C2cEf1b0D44,
    0x73bec2D5C98BC4CacE92c72C5aB55eea1bc39aCC,
    0x91E11605370356E2576778B0f56a07C3b3d09DF2,
    0x0729266a0E42204546edAc4CAf53b47dbB620D30,
    0xB0D3d8338FdE2b781CB5dA295a20C2b31ACCCb6c,
    0x8a9c644A2c7c460D3FA8716CfEAC5Cb21395D4e1
  ];
  uint256[] public amounts =   [
    387998023671410000000,
    375278689963090000000,
    352018257065290000000,
    332189924227500000000,
    322751130934770000000,
    266636326056720000000,
    252830792367340000000,
    249895090713520000000,
    247090173958270000000,
    219087530879770000000,
    212499662643820000000,
    210395705587940000000,
    209488901593770000000,
    208291748532060000000,
    208143209163910000000,
    205807357302090000000,
    203773527877870000000,
    202370569995870000000,
    194485224897950000000,
    166596727142350000000,
    163265586206090000000,
    153915431947540000000,
    130546826367650000000,
    118830472412650000000,
    114746282689790000000,
    102954239149280000000,
    97242612448980000000,
    97242612448980000000,
    88220695696750000000,
    80241961305560000000,
    71545852761740000000,
    64709458478980000000,
    63118711676380000000,
    63118711676380000000,
    62709854813140000000,
    62422556835680000000,
    58345567469390000000,
    48621306224490000000,
    45256711833750000000,
    42079141117590000000,
    39083089545730000000,
    32868182554190000000,
    28665528215490000000,
    26873112641410000000,
    26768102178810000000,
    24623132523090000000,
    23712375748650000000,
    21516742062320000000,
    20377352787790000000,
    19022774883970000000,
    18935613502910000000,
    18367378448190000000,
    16340608704300000000,
    14977129125280000000,
    4270237122490000000,
    4064741200370000000,
    2130277022560000000,
    2665486175308850000000
  ];
  uint256 public total = 9764845721712500000000;

  function proceed() public onlyOwner {
    // Start giving errors if we're not done.
    require(!complete);
    token.mint(this, total);
    for(uint256 i = 0; i < holders.length; i++) {
      token.transfer(holders[i], amounts[i]);
    }
    returnOwnership();
    complete = true;
  }

  function setToken(MintableObject _token) public onlyOwner {
    token = _token;
  }

  function returnOwnership() public onlyOwner {
    token.transferOwnership(owner);
  }
}
