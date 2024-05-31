pragma solidity 0.4.25;

/* This Source Code Form is subject to the terms of the Mozilla external
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * This code has not been reviewed.
 * Do not use or deploy this code before reviewing it personally first.
 */

interface ERC777Token {
  function name() external view returns (string);
  function symbol() external view returns (string);
  function totalSupply() external view returns (uint256);
  function balanceOf(address owner) external view returns (uint256);
  function granularity() external view returns (uint256);

  function defaultOperators() external view returns (address[]);
  function isOperatorFor(address operator, address tokenHolder) external view returns (bool);
  function authorizeOperator(address operator) external;
  function revokeOperator(address operator) external;

  function send(address to, uint256 amount, bytes holderData) external;
  function operatorSend(address from, address to, uint256 amount, bytes holderData, bytes operatorData) external;

  function burn(uint256 amount, bytes holderData) external;
  function operatorBurn(address from, uint256 amount, bytes holderData, bytes operatorData) external;

  event Sent(
    address indexed operator,
    address indexed from,
    address indexed to,
    uint256 amount,
    bytes holderData,
    bytes operatorData
  );
  event Minted(address indexed operator, address indexed to, uint256 amount, bytes operatorData);
  event Burned(address indexed operator, address indexed from, uint256 amount, bytes holderData, bytes operatorData);
  event AuthorizedOperator(address indexed operator, address indexed tokenHolder);
  event RevokedOperator(address indexed operator, address indexed tokenHolder);
}

/// @title DelegatedTransferOperatorV4
/// @author Roger Wu (Roger-Wu)
/// @dev A DelegatedTransferOperator contract that has the following features:
///   1. To prevent replay attack, we check if a _nonce has been used by a token holder.
///   2. Minimize the gas by making functions inline and remove trivial event.
///   3. Add _userData.
contract DelegatedTransferOperatorV4 {
  mapping(address => uint256) public usedNonce;
  ERC777Token public tokenContract;

  constructor(address _tokenAddress) public {
    tokenContract = ERC777Token(_tokenAddress);
  }

  /**
    * @notice Submit a presigned transfer
    * @param _to address The address which you want to transfer to.
    * @param _delegate address The address which is allowed to send this transaction.
    * @param _value uint256 The amount of tokens to be transferred.
    * @param _fee uint256 The amount of tokens paid to msg.sender, by the owner.
    * @param _nonce uint256 Presigned transaction number.
    * @param _userData bytes Data generated by the user to be sent to the recipient.
    * @param _sig_r bytes32 The r of the signature.
    * @param _sig_s bytes32 The s of the signature.
    * @param _sig_v uint8 The v of the signature.
    * @notice some rules:
    * 1. If _to is address(0), the tx will fail when doSend().
    * 2. If _delegate == address(0), then anyone can be the delegate.
    * 3. _nonce must be greater than the last used nonce by the token holder,
    *    but nonces don't have to be serial numbers.
    *    We recommend using unix time as nonce.
    * 4. _sig_v should be 27 or 28.
    */
  function transferPreSigned(
    address _to,
    address _delegate,
    uint256 _value,
    uint256 _fee,
    uint256 _nonce,
    bytes _userData,
    bytes32 _sig_r,
    bytes32 _sig_s,
    uint8 _sig_v
  )
    external
  {
    require(
      _delegate == address(0) || _delegate == msg.sender,
      "_delegate should be address(0) or msg.sender"
    );

    // address _signer = recover(_hash, _signature);
    address _signer = (_sig_v != 27 && _sig_v != 28) ?
      address(0) :
      ecrecover(
        keccak256(abi.encodePacked(
          address(this),
          _to,
          _delegate,
          _value,
          _fee,
          _nonce,
          _userData
        )),
        _sig_v, _sig_r, _sig_s
      );

    require(
      _signer != address(0),
      "_signature is invalid."
    );

    require(
      _nonce > usedNonce[_signer],
      "_nonce must be greater than the last used nonce of the token holder."
    );

    usedNonce[_signer] = _nonce;

    tokenContract.operatorSend(_signer, _to, _value, _userData, "");
    if (_fee > 0) {
      tokenContract.operatorSend(_signer, msg.sender, _fee, _userData, "");
    }
  }

  /**
    * @notice Hash (keccak256) of the payload used by transferPreSigned
    * @param _operator address The address of the operator.
    * @param _to address The address which you want to transfer to.
    * @param _delegate address The address of the delegate.
    * @param _value uint256 The amount of tokens to be transferred.
    * @param _fee uint256 The amount of tokens paid to msg.sender, by the owner.
    * @param _nonce uint256 Presigned transaction number.
    * @param _userData bytes Data generated by the user to be sent to the recipient.
    */
  function transferPreSignedHashing(
    address _operator,
    address _to,
    address _delegate,
    uint256 _value,
    uint256 _fee,
    uint256 _nonce,
    bytes _userData
  )
    public
    pure
    returns (bytes32)
  {
    return keccak256(abi.encodePacked(
      _operator,
      _to,
      _delegate,
      _value,
      _fee,
      _nonce,
      _userData
    ));
  }

  /**
    * @notice Recover signer address from a message by using his signature
    * @param hash bytes32 message, the hash is the signed message. What is recovered is the signer address.
    * @param sig bytes signature, the signature is generated using web3.eth.sign()
    */
  function recover(bytes32 hash, bytes sig) public pure returns (address) {
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
      r := mload(add(sig, 0x20))
      s := mload(add(sig, 0x40))
      v := byte(0, mload(add(sig, 0x60)))
    }

    // Version of signature should be 27 or 28, but 0 and 1 are also possible versions
    if (v < 27) {
      v += 27;
    }

    // If the version is correct return the signer address
    if (v != 27 && v != 28) {
      return (address(0));
    } else {
      return ecrecover(hash, v, r, s);
    }
  }
}
