pragma solidity ^0.4.10;

contract addGenesisPairs {
    
address[] newParents;
address[] newChildren;

function addGenesisPairs()    {
    // Set elixor contract address
    elixor elixorContract=elixor(0x898bF39cd67658bd63577fB00A2A3571dAecbC53);
    
    newParents=[
0xc41A79EE7BA108BF58164ef58d5fc420EC6392De,
0x5c7Cf854200c7eAFFC0D648Ad4E44Ff6f354Fb26,
0xFdaC5A83aaEC70853730183464b9FE0037F9921c,
0xEdd154655222A8813C26e7139C634c1446b9EF31,
0x614ef0A188b5c499954E9178E6cb6DbE816cE7f5,
0xD5290aEde47D6198E384d293f1799a74d8895DaF,
0x00B60C459c77AA31B870b6d9b0CA82C96e3c00Cf,
0x99408F26e2f4547dCaa42Df755B05427f45FdD90,
0x29484cE5F5bB7f6607eAD5A451F4C558B09eEf25,
0xBCcB56857495801a0eCBaF0d2C81E31204330170,
0xbcc204CDC2CBd8b04D2c4fb37D942801efc8693A,
0xf38e87c415715813f5384Cd316Cba4c225f42bb9,
0x48b3A08a5Bf41221Eeb953504E30Ec984a89fA05,
0x0b6A43bF665715B5B0D105AA6d9a0dd50B712B85,
0x3172Bb0011689395691E76fC4384727B5d6092DE,
0x763f958e42578b9646D4BFab8112fDe84adaF258,
0x834c947E095A91e14D589589597977C25E402cF9,
0x128EA00b124AA228f4f18adA59fC2B017BdabF31,
0xdFe01DB72a340B328Bb93218D5Dc4c466005Ec21,
0xAA64Ff9209D197aC68ec8D6C5F235f3541806cF9,
0x6F7b0ACdDEb02E433cC17786616fa4b9eeaB34A9,
0x453074A3E095f9a596671A1188db8CDd51e404e0,
0xF46F362f336c5e3AA1667D7e826ed7525fc3686F,
0xA6CfBb051a3C6EEb27324dDd5dC38484c018E250,
0x87ea5e20c9d2eE2cC4c3F02eF6A0BCB16e0CE925,
0x785a5D5e3B91Cc983f079f7800bEc1a99cE446F2,
0xCbc9bC577E179107709b73Dd6192447C9bC52dCf,
0xb71D1c3dd6A12d1947C20dd3629E396776DaC46d,
0x860b0a4ee39b80cBF6961bd2260E6c798Ff64C27
    ];
    
    
    
    newChildren=[0x6dd50cBCC0F0D94dF60A15a7D6C621B2AC067614,
0xfBdB1BC6FAfbE479Dc97EC99fD507642486Efe5d,
0x1f15628333116d6E3F409e716469a0976aD56684,
0x27F5C742888Aa7a31DC4290F86ED4D29794345E4,
0x67dF42A859Cf4a8A25C6837080e7eF8b92DC3148,
0x16A77AeD414eD19F939E231Ae2c528b19563c659,
0x00D8fA39537D987a462506d878E80169Cb6cdFE4,
0x3555c3493E2486Db877e6AE150a1E044B82618Cf,
0xE73701E2Df1233c17911612925367dB3a893731E,
0x16c73E8Cbf1725c5233CC96E526b539bF0838789,
0x9C11aB48bf5ecd357A00E5B9595B3f980dfd30A6,
0xd35586cc58883AA6233b09a12C5c9ba242018F94,
0x994F37fA81d94Ecf51CE246Ba1589Ec079804789,
0x146D030cB383Cb536b4FC60A825F3922a04faf32,
0x380513CFBc39a87B5678C6759987bD8B69DaC6F8,
0x497E5C6b2e8B498Ce1393f74Bb06E79b2728aFE1,
0x72F876F21db3B2355dfe4D7F3591CB9F0604E636,
0xc901f6a82ed26F52687D2e415E95D4d721c51759,
0x60331E0272E45883f16eC164DF6c85baC5aDc356,
0xE8c001896DacDB71A84eBE081A0dd1f9cf275F25,
0x00b7EA3b9d746A6024FEC7D4e2b25cbA18549A51,
0x27f30968934aAAe8032c34d8C6A1D3F3A51814d5,
0xca9030805B835e6EcBFC7c8D40d06E18768569c7,
0x7fB1DAe1Adb45A26A1e397E5b1Fe801c4e504Cb5,
0x0aeD67c456D0Ab9757A0D9B06F48d47436040F33,
0x18105841EBaa74F760cDE0bc1f7E2cc98d9d924f,
0x0B56bb4B7E91e440813CdDeD8adF9c92798Ec3B7,
0x072Ed7Cd94Bc40cc21ebb8eBB406bC785Da31B3b,
0xf5110b9Fe182130019262908bc00f60CF4130216
    ];
    
    elixorContract.importGenesisPairs(newParents,newChildren);
 
}

}

contract elixor {
    function importGenesisPairs(address[] newParents,address[] newChildren) public;
}
