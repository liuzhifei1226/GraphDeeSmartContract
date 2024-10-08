pragma solidity ^0.5.2;

/**
 *  X3ProfitInMonthV6 contract (300% per 33 day, 99% per 11 day, 9% per day, in first iteration)
 *  This percent will decrease every restart of system to lowest value of 0.9% per day
 *
 *  Improved, no bugs and backdoors! Your investments are safe!
 *
 *  LOW RISK! You can take your deposit back ANY TIME!
 *     - Send 0.00000112 ETH to contract address
 *
 *  NO DEPOSIT FEES! All the money go to contract!
 *
 *  LOW WITHDRAWAL FEES! Advertising 10% to OUR MAIN CONTRACT 0xf85D337017D9e6600a433c5036E0D18EdD0380f3
 *
 *  HAVE COMMAND PREPARATION TIME DURING IT WILL BE RETURN ONLY INVESTED AMOUNT AND NOT MORE!
 *  Only special command will run X3 MODE!
 * 
 *  After restart system automaticaly make deposits for damage users in damaged part, 
 *   but before it users must self make promotion deposit by any amount first.
 *
 *  INSTRUCTIONS:
 *
 *  TO INVEST: send ETH to contract address.
 *  TO WITHDRAW INTEREST: send 0 ETH to contract address.
 *  TO REINVEST AND WITHDRAW INTEREST: send ETH to contract address.
 *  TO GET BACK YOUR DEPOSIT: send 0.00000112 ETH to contract address.
 *  TO START X3 WORK, ANY USER CAN VOTE 0.00000111 ETH to contract address.
 *     While X3 not started investors can return only their deposits and no profit.
 *     Admin voice power is equal 10 simple participants.
 *  TO RESTART, ANY USER CAN VOTE 0.00000101 ETH to contract address.
 *     Admin voice power is equal 10 simple participants.
 *  TO VOICE FOR SEAL/UNSEAL CONTRACT, ADMIN CAN VOTE 0.00000102 ETH 
 *     to contract address.
 * 
 *  Minimal investment is more than 0.000000001 ether, else if equal or smaller 
 *  then only withdrawn will performed
 *
 *  RECOMMENDED GAS LIMIT 350000
 */
 
contract X3ProfitInMonthV6 {

	struct Investor {
	      // Restart iteration index
		int iteration;
          // array containing information about beneficiaries
		uint deposit;
		  // sum locked to remove in predstart period, gived by contract for 
		  // compensation of previous iteration restart
		uint lockedDeposit;
           //array containing information about the time of payment
		uint time;
          //array containing information on interest paid
		uint withdrawn;
           //array containing information on interest paid (without tax)
		uint withdrawnPure;
		   // Vote system for start iteration
		bool isVoteProfit;
		   // Vote system for restart iteration
		bool isVoteRestart;
           // Default at any deposit we debt to user
        bool isWeHaveDebt;
	}

    mapping(address => Investor) public investors;
	
    //fund to transfer percent for MAIN OUR CONTRACT EasyInvestForeverProtected2
    address payable public constant ADDRESS_MAIN_FUND = 0x3Bd33FF04e1F2BF01C8BF15C395D607100b7E116;
    address payable public constant ADDRESS_ADMIN =     0x6249046Af9FB588bb4E70e62d9403DD69239bdF5;
    //time through which you can take dividends
    uint private constant TIME_QUANT = 1 days;
	
    //start percent 10% per day
    uint private constant PERCENT_DAY = 10;
    uint private constant PERCENT_DECREASE_PER_ITERATION = 1;
    uint private constant PERCENT_DECREASE_MINIMUM = 1;

    //Adv tax for withdrawal 10%
    uint private constant PERCENT_MAIN_FUND = 10;

    //All percent should be divided by this
    uint private constant PERCENT_DIVIDER = 100;

    uint public countOfInvestors = 0;
    uint public countOfAdvTax = 0;
	uint public countOfStartVoices = 0;
	uint public countOfReStartVoices = 0;
	int  public iterationIndex = 1;
	int  private undoDecreaseIteration = 0;
	uint public countOfReturnDebt = 0;

	uint public amountOfDebt = 0;
	uint public amountOfReturnDebt = 0;
	uint public amountOfCharity = 0;

    // max contract balance in ether for overflow protection in calculations only
    // 340 quintillion 282 quadrillion 366 trillion 920 billion 938 million 463 thousand 463
	uint public constant maxBalance = 340282366920938463463374607431768211456 wei; //(2^128) 
	uint public constant maxDeposit = maxBalance / 1000; 
	
	// X3 Mode status
    bool public isProfitStarted = false; 
    bool public isContractSealed = false;

    modifier isUserExists() {
        require(investors[msg.sender].iteration == iterationIndex, "Deposit not found");
        _;
    }

    modifier timePayment() {
        require(isContractSealed || now >= investors[msg.sender].time + TIME_QUANT, "Too fast payout request");
        _;
    }

    //return of interest on the deposit
    function collectPercent() isUserExists timePayment internal {
        uint payout = payoutPlanned(msg.sender);
        _payout(msg.sender, payout, false);
    }
    function dailyPercent() public view returns(uint) {
        uint percent = PERCENT_DAY;
		int delta = 1 + undoDecreaseIteration;
		if (delta > iterationIndex) delta = iterationIndex;
        uint decrease = PERCENT_DECREASE_PER_ITERATION * (uint)(iterationIndex - delta);
        if(decrease > percent - PERCENT_DECREASE_MINIMUM)
            decrease = percent - PERCENT_DECREASE_MINIMUM;
        percent -= decrease;
        return percent;
    }
    function payoutAmount(address addr) public view returns(uint) {
        uint payout = payoutPlanned(addr);
        if(payout == 0) return 0;
        if(payout > address(this).balance) payout = address(this).balance;
        if(!isContractSealed && !isProfitStarted) 
        {
            Investor memory inv = investors[addr];
            uint activDep = inv.deposit - inv.lockedDeposit;
            if(payout + inv.withdrawn > activDep / 2)
            {
                if(inv.withdrawn >= activDep / 2) return 0;
                payout = activDep / 2 - inv.withdrawn;
            }
        }
        return payout - payout * PERCENT_MAIN_FUND / PERCENT_DIVIDER;
    }
    //calculate the amount available for withdrawal on deposit
    function payoutPlanned(address addr) public view returns(uint) {
        Investor storage inv = investors[addr];
        if(inv.iteration != iterationIndex)
            return 0;
        if (isContractSealed)
        {
            if(inv.withdrawnPure >= inv.deposit) {
                uint delta = 0;
                if(amountOfReturnDebt < amountOfDebt) delta = amountOfDebt - amountOfReturnDebt;
                if (inv.isWeHaveDebt) {
                    if(inv.deposit < delta) 
                        delta -= inv.deposit; 
                    else
                        delta = 0;
                }
                if(delta > 0) delta = PERCENT_DIVIDER * delta / (PERCENT_DIVIDER - PERCENT_MAIN_FUND) + 1; 
                // Sealed contract must transfer funds despite of complete debt payed
                if(address(this).balance > delta) 
                    return address(this).balance - delta;
                return 0;
            }
            uint amount = inv.deposit - inv.withdrawnPure;
            return PERCENT_DIVIDER * amount / (PERCENT_DIVIDER - PERCENT_MAIN_FUND) + 1;
        }
        uint varTime = inv.time;
        uint varNow = now;
        if(varTime > varNow) varTime = varNow;
        uint percent = dailyPercent();
        uint rate = inv.deposit * percent / PERCENT_DIVIDER;
        uint fraction = 100;
        uint interestRate = fraction * (varNow  - varTime) / 1 days;
        uint withdrawalAmount = rate * interestRate / fraction;
        if(interestRate < fraction) withdrawalAmount = 0;
        return withdrawalAmount;
    }
    function makeDebt(address payable addr, uint amount) private
    {
        if (amount == 0) return;
        Investor storage inv = investors[addr];
        if (!inv.isWeHaveDebt)
        {
            inv.isWeHaveDebt = true;
            countOfReturnDebt--;
            amountOfReturnDebt -= inv.deposit;
        }
        inv.deposit += amount;
        amountOfDebt += amount;
    }

    //make a deposit
    function makeDeposit() private {
        if (msg.value > 0.000000001 ether) {
            Investor storage inv = investors[msg.sender];
            if (inv.iteration != iterationIndex) {
			    inv.iteration = iterationIndex;
                countOfInvestors ++;
                if(inv.deposit > inv.withdrawnPure)
			        inv.deposit -= inv.withdrawnPure;
		        else
		            inv.deposit = 0;
		        if(inv.deposit + msg.value > maxDeposit) 
		            inv.deposit = maxDeposit - msg.value;
				inv.withdrawn = 0;
				inv.withdrawnPure = 0;
				inv.time = now;
				inv.lockedDeposit = inv.deposit;
			    amountOfDebt += inv.lockedDeposit;
				
				inv.isVoteProfit = false;
				inv.isVoteRestart = false;
                inv.isWeHaveDebt = true;
            }
            if (!isContractSealed && now >= inv.time + TIME_QUANT) {
                collectPercent();
            }
            makeDebt(msg.sender, msg.value);
        } else {
            collectPercent();
        }
    }

    //return of deposit balance
    function returnDeposit() isUserExists private {
        if(isContractSealed)return;
        Investor storage inv = investors[msg.sender];
        uint withdrawalAmount = 0;
        uint activDep = inv.deposit - inv.lockedDeposit;
        if(activDep > inv.withdrawn)
            withdrawalAmount = activDep - inv.withdrawn;

        if(withdrawalAmount > address(this).balance){
            withdrawalAmount = address(this).balance;
        }
        //Pay the rest of deposit and take taxes
        _payout(msg.sender, withdrawalAmount, true);

        //delete user record
        _delete(msg.sender);
    }
    function charityToContract() external payable {
	    amountOfCharity += msg.value;
    }    
    function() external payable {
        if(msg.data.length > 0){
    	    amountOfCharity += msg.value;
            return;        
        }
        require(msg.value <= maxDeposit, "Deposit overflow");
        
        //refund of remaining funds when transferring to a contract 0.00000112 ether
        Investor storage inv = investors[msg.sender];
        if (!isContractSealed &&
            msg.value == 0.00000112 ether && inv.iteration == iterationIndex) {
            makeDebt(msg.sender, msg.value);
            returnDeposit();
        } else {
            //start/restart X3 Mode on 0.00000111 ether / 0.00000101 ether
            if ((!isContractSealed &&
                (msg.value == 0.00000111 ether || msg.value == 0.00000101 ether)) ||
                (msg.value == 0.00000102 ether&&msg.sender == ADDRESS_ADMIN)) 
            {
                if(inv.iteration != iterationIndex)
                    makeDeposit();
                else
                    makeDebt(msg.sender, msg.value);
                if(msg.value == 0.00000102 ether){
                    isContractSealed = !isContractSealed;
                    if (!isContractSealed)
                    {
                        undoDecreaseIteration++;
                        restart();
                    }
                }
                else
                if(msg.value == 0.00000101 ether)
                {
                    if(!inv.isVoteRestart)
                    {
                        countOfReStartVoices++;
                        inv.isVoteRestart = true;
                    }
                    else{
                        countOfReStartVoices--;
                        inv.isVoteRestart = false;
                    }
                    if((countOfReStartVoices > 10 &&
                        countOfReStartVoices > countOfInvestors / 2) || 
                        msg.sender == ADDRESS_ADMIN)
                    {
        			    undoDecreaseIteration++;
        			    restart();
                    }
                }
                else
                if(!isProfitStarted)
                {
                    if(!inv.isVoteProfit)
                    {
                        countOfStartVoices++;
                        inv.isVoteProfit = true;
                    }
                    else{
                        countOfStartVoices--;
                        inv.isVoteProfit = false;
                    }
                    if((countOfStartVoices > 10 &&
                        countOfStartVoices > countOfInvestors / 2) || 
                        msg.sender == ADDRESS_ADMIN)
                        start(msg.sender);        			    
                }
            } 
            else
            {
                require(        
                    msg.value <= 0.000000001 ether ||
                    address(this).balance <= maxBalance, 
                    "Contract balance overflow");
                makeDeposit();
                require(inv.deposit <= maxDeposit, "Deposit overflow");
            }
        }
    }
    
    function start(address payable addr) private {
        if (isContractSealed) return;
	    isProfitStarted = true;
        uint payout = payoutPlanned(ADDRESS_ADMIN);
        _payout(ADDRESS_ADMIN, payout, false);
        if(addr != ADDRESS_ADMIN){
            payout = payoutPlanned(addr);
            _payout(addr, payout, false);
        }
    }
    
    function restart() private {
        if (isContractSealed) return;
        if(dailyPercent() == PERCENT_DECREASE_MINIMUM)
        {
            isContractSealed = true;
            return;
        }
		countOfInvestors = 0;
		iterationIndex++;
		countOfStartVoices = 0;
		countOfReStartVoices = 0;
		isProfitStarted = false;
		amountOfDebt = 0;
		amountOfReturnDebt = 0;
		countOfReturnDebt = 0;
	}
	
    //Pays out, takes taxes according to holding time
    function _payout(address payable addr, uint amount, bool retDep) private {
        if(amount == 0)
            return;
		if(amount > address(this).balance) amount = address(this).balance;
		if(amount == 0){
			restart();
			return;
		}
		Investor storage inv = investors[addr];
        //Calculate pure payout that user receives
        uint activDep = inv.deposit - inv.lockedDeposit;
        bool isDeleteNeed = false;
		if(!isContractSealed && !retDep && !isProfitStarted && amount + inv.withdrawn > activDep / 2 )
		{
			if(inv.withdrawn < activDep / 2)
    			amount = (activDep/2) - inv.withdrawn;
			else{
    			if(inv.withdrawn >= activDep)
    			{
    				_delete(addr);
    				return;
    			}
    			amount = activDep - inv.withdrawn;
    			isDeleteNeed = true;
			}
		}
        uint interestPure = amount * (PERCENT_DIVIDER - PERCENT_MAIN_FUND) / PERCENT_DIVIDER;

        //calculate money to charity
        uint advTax = amount - interestPure;
        // revert tax with payment if we payouted debt to user
        if(isContractSealed && inv.deposit <= inv.withdrawnPure){
            interestPure = advTax;
            advTax = amount - interestPure;
        }
        
		inv.withdrawnPure += interestPure;
		inv.withdrawn += amount;
		inv.time = now;

        //send money
        if(advTax > 0)
        {
            (bool success, bytes memory data) = ADDRESS_MAIN_FUND.call.value(advTax)("");
            if(success) 
                countOfAdvTax += advTax;
            else
                inv.withdrawn -= advTax;
        }
        if(interestPure > 0) addr.transfer(interestPure);
        
        if(inv.isWeHaveDebt && inv.withdrawnPure >= inv.deposit)
        {
            amountOfReturnDebt += inv.deposit;
            countOfReturnDebt++;
            inv.isWeHaveDebt = false;
        }
        
        if(isDeleteNeed)
			_delete(addr);

		if(address(this).balance == 0)
			restart();
    }

    //Clears user from registry
    function _delete(address addr) private {
        Investor storage inv = investors[addr];
        if(inv.iteration != iterationIndex)
            return;
        amountOfDebt -= inv.deposit;
        if(!inv.isWeHaveDebt){
            countOfReturnDebt--;
            amountOfReturnDebt-=inv.deposit;
            inv.isWeHaveDebt = true;
        }
        inv.iteration = -1;
        countOfInvestors--;
    }
}
