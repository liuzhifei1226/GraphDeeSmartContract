pragma solidity ^0.4.25;

/**

  EN:

  Web: https://www.multy150.today/
  Telegram: https://t.me/multy150today

  Queue contract: returns 150% of each investment!

  Automatic payouts!
  No bugs, no backdoors, NO OWNER - fully automatic!
  Made and checked by professionals!

  1. Send any sum to smart contract address
     - sum from 0.05 ETH
     - min 350 000 gas limit
     - you are added to a queue
  2. Wait a little bit
  3. ...
  4. PROFIT! You have got 150%

  How is that?
  1. The first investor in the queue (you will become the
     first in some time) receives next investments until
     it become 150% of his initial investment.
  2. You will receive payments in several parts or all at once
  3. Once you receive 150% of your initial investment you are
     removed from the queue.
  4. The balance of this contract should normally be 0 because
     all the money are immediately go to payouts


     So the last pays to the first (or to several first ones
     if the deposit big enough) and the investors paid 150% are removed from the queue

                new investor --|               brand new investor --|
                 investor5     |                 new investor       |
                 investor4     |     =======>      investor5        |
                 investor3     |                   investor4        |
    (part. paid) investor2    <|                   investor3        |
    (fully paid) investor1   <-|                   investor2   <----|  (pay until 150%)

    ==> Limits: <==

    Multiplier: 150%
    Minimum deposit: 0.05ETH
    Maximum deposit: 10ETH
*/


/**

  RU:

  Web: https://www.multy150.today/
  Telegram: https://t.me/multy150today

  Контракт Умная Очередь: возвращает 150% от вашего депозита!

  Автоматические выплаты!
  Без ошибок, дыр, автоматический - для выплат НЕ НУЖНА администрация!
  Создан и проверен профессионалами!

  1. Пошлите любую ненулевую сумму на адрес контракта
     - сумма от 0.05 ETH
     - gas limit минимум 350 000
     - вы встанете в очередь
  2. Немного подождите
  3. ...
  4. PROFIT! Вам пришло 150% от вашего депозита.

  Как это возможно?
  1. Первый инвестор в очереди (вы станете первым очень скоро) получает выплаты от
     новых инвесторов до тех пор, пока не получит 150% от своего депозита
  2. Выплаты могут приходить несколькими частями или все сразу
  3. Как только вы получаете 150% от вашего депозита, вы удаляетесь из очереди
  4. Баланс этого контракта должен обычно быть в районе 0, потому что все поступления
     сразу же направляются на выплаты

     Таким образом, последние платят первым, и инвесторы, достигшие выплат 150% от депозита,
     удаляются из очереди, уступая место остальным

              новый инвестор --|            совсем новый инвестор --|
                 инвестор5     |                новый инвестор      |
                 инвестор4     |     =======>      инвестор5        |
                 инвестор3     |                   инвестор4        |
 (част. выплата) инвестор2    <|                   инвестор3        |
(полная выплата) инвестор1   <-|                   инвестор2   <----|  (доплата до 150%)

    ==> Лимиты: <==

    Профит: 150%
    Минимальный вклад: 0.05 ETH
    Максимальный вклад: 10 ETH


*/
contract Multy {

	//Address for promo expences
    address constant private PROMO = 0xa3093FdE89050b3EAF6A9705f343757b4DfDCc4d;
	address constant private PRIZE = 0x86C1185CE646e549B13A6675C7a1DF073f3E3c0A;
	
	//Percent for promo expences
    uint constant public PROMO_PERCENT = 6;
    
    //Bonus prize
    uint constant public BONUS_PERCENT = 4;
		
    //The deposit structure holds all the info about the deposit made
    struct Deposit {
        address depositor; // The depositor address
        uint deposit;   // The deposit amount
        uint payout; // Amount already paid
    }

    Deposit[] public queue;  // The queue
    mapping (address => uint) public depositNumber; // investor deposit index
    uint public currentReceiverIndex; // The index of the depositor in the queue
    uint public totalInvested; // Total invested amount

    //This function receives all the deposits
    //stores them and make immediate payouts
    function () public payable {
        
        require(block.number >= 6655835);

        if(msg.value > 0){

            require(gasleft() >= 250000); // We need gas to process queue
            require(msg.value >= 0.05 ether && msg.value <= 10 ether); // Too small and too big deposits are not accepted
            
            // Add the investor into the queue
            queue.push( Deposit(msg.sender, msg.value, 0) );
            depositNumber[msg.sender] = queue.length;

            totalInvested += msg.value;

            //Send some promo to enable queue contracts to leave long-long time
            uint promo = msg.value*PROMO_PERCENT/100;
            PROMO.send(promo);
            uint prize = msg.value*BONUS_PERCENT/100;
            PRIZE.send(prize);
            
            // Pay to first investors in line
            pay();

        }
    }

    // Used to pay to current investors
    // Each new transaction processes 1 - 4+ investors in the head of queue
    // depending on balance and gas left
    function pay() internal {

        uint money = address(this).balance;
        uint multiplier = 150;

        // We will do cycle on the queue
        for (uint i = 0; i < queue.length; i++){

            uint idx = currentReceiverIndex + i;  //get the index of the currently first investor

            Deposit storage dep = queue[idx]; // get the info of the first investor

            uint totalPayout = dep.deposit * multiplier / 100;
            uint leftPayout;

            if (totalPayout > dep.payout) {
                leftPayout = totalPayout - dep.payout;
            }

            if (money >= leftPayout) { //If we have enough money on the contract to fully pay to investor

                if (leftPayout > 0) {
                    dep.depositor.send(leftPayout); // Send money to him
                    money -= leftPayout;
                }

                // this investor is fully paid, so remove him
                depositNumber[dep.depositor] = 0;
                delete queue[idx];

            } else{

                // Here we don't have enough money so partially pay to investor
                dep.depositor.send(money); // Send to him everything we have
                dep.payout += money;       // Update the payout amount
                break;                     // Exit cycle

            }

            if (gasleft() <= 55000) {         // Check the gas left. If it is low, exit the cycle
                break;                       // The next investor will process the line further
            }
        }

        currentReceiverIndex += i; //Update the index of the current first investor
    }
    
    //Returns your position in queue
    function getDepositsCount(address depositor) public view returns (uint) {
        uint c = 0;
        for(uint i=currentReceiverIndex; i<queue.length; ++i){
            if(queue[i].depositor == depositor)
                c++;
        }
        return c;
    }

    // Get current queue size
    function getQueueLength() public view returns (uint) {
        return queue.length - currentReceiverIndex;
    }

}
